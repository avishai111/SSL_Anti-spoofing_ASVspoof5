import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, pad
from model import Model
from core_scripts.startup_config import set_random_seed
import os
import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from tqdm import tqdm
import librosa
from model import *
import copy


class WrappedModel(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]


        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, return_feat = True):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)
        
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        if return_feat:
            return output, x_ssl_feat

        return output


def genSpoof_list(dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             speaker,key,gender,_,_,_,_,attack,label,_ = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            speaker,key,gender,_,_,_,_,attack,label,_ = line.strip().split()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             speaker,key,gender,_,_,_,_,attack,label,_ = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out, x_ssl_feat = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path, normalize):
    # אפשר לשנות את batch_size ו-num_workers לפי הזיכרון שלך
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader))

    ssl_feats = {}
    chunk_size = 5000        # number of utterances per .npz (tune this)
    num_in_chunk = 0
    chunk_idx = 0

    with torch.no_grad(), open(save_path, 'w') as fh:
        for batch_x, utt_id in pbar:
            batch_x = batch_x.to(device, non_blocking=True)

            batch_out, x_ssl_feat = model(batch_x)
            batch_score = batch_out[:, 1].detach().cpu().numpy().ravel()

            pbar.set_description(f"Processing: {utt_id[0]}")
            pbar.set_postfix(score=float(batch_score[0]))

            for f, cm in zip(utt_id, batch_score):
                fh.write(f"{f} {cm}\n")

            x_np = x_ssl_feat.detach().cpu().numpy()

            for i, f in enumerate(utt_id):
                ssl_feats[f] = x_np[i]
                num_in_chunk += 1

                # when chunk is full, flush to disk and reset
                if num_in_chunk >= chunk_size:
                    if normalize:
                        fname = f"all_ssl_features_normalize_part{chunk_idx}.npz"
                    else:
                        fname = f"all_ssl_features_no_normalize_part{chunk_idx}.npz"
                    np.savez(fname, **ssl_feats)
                    ssl_feats = {}
                    num_in_chunk = 0
                    chunk_idx += 1

    # flush any remaining features
    if ssl_feats:
        if normalize:
            fname = f"all_ssl_features_normalize_part{chunk_idx}.npz"
        else:
            fname = f"all_ssl_features_no_normalize_part{chunk_idx}.npz"
        np.savez(fname, **ssl_feats)

    print(f"Scores saved to {save_path}")
    print(f"Saved {chunk_idx + (1 if num_in_chunk > 0 else 0)} feature files in chunks")




def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, xfeat_ssl = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss


class Dataset_ASVspoof05_eval(Dataset):
	def __init__(self, list_IDs, base_dir, normalize):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=16000*4 # take ~4 sec audio (64600 samples)
            self.normalize = normalize

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'/'+utt_id+'.flac', sr=16000)
            if self.normalize:
                peak = np.max(np.abs(X))
                X = X / peak
            X = pad(X, self.cut)
            X_inp = torch.from_numpy(X).float() 
            return X_inp,utt_id  


def run_asvspoof2021_baseline(
    database_path='/gpfs0/bgu-benshimo/projects/ASVspoof5/',
    protocols_path='/gpfs0/bgu-benshimo/projects/ASVspoof5/ASVspoof5_protocols/',
    batch_size=1,
    num_epochs=100,
    lr=0.000001,
    weight_decay=0.0001,
    loss='weighted_CCE',
    seed=1234,
    model_path='/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/models/Best_LA_model_for_DF.pth',
    comment=None,
    track='DF',
    eval_output='/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/',
    eval_mode=True,
    is_eval=True,
    eval_part=0,
    cudnn_deterministic_toggle=True,
    cudnn_benchmark_toggle=False,
    algo=3,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=10,
    maxCoeff=100,
    minG=0,
    maxG=0,
    minBiasLinNonLin=5,
    maxBiasLinNonLin=20,
    N_f=5,
    P=10,
    g_sd=2,
    SNRmin=10,
    SNRmax=40,
):
    normalize = True

    if normalize == True:
        eval_output = '/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/train_asvspoof5_normalize.txt'
    else:
        eval_output = '/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/train_asvspoof5_no_normalize.txt'

    args = SimpleNamespace(
        database_path=database_path,
        protocols_path=protocols_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        loss=loss,
        seed=seed,
        model_path=model_path,
        comment=comment,
        track=track,
        eval_output=eval_output,
        eval=eval_mode,
        is_eval=is_eval,
        eval_part=eval_part,
        cudnn_deterministic_toggle=cudnn_deterministic_toggle,
        cudnn_benchmark_toggle=cudnn_benchmark_toggle,
        algo=algo,
        nBands=nBands,
        minF=minF,
        maxF=maxF,
        minBW=minBW,
        maxBW=maxBW,
        minCoeff=minCoeff,
        maxCoeff=maxCoeff,
        minG=minG,
        maxG=maxG,
        minBiasLinNonLin=minBiasLinNonLin,
        maxBiasLinNonLin=maxBiasLinNonLin,
        N_f=N_f,
        P=P,
        g_sd=g_sd,
        SNRmin=SNRmin,
        SNRmax=SNRmax,
    )

    ASVspoof_path_name = 'ASVspoof5.train.tsv'
    folder_files = 'flac_T' 
    if not os.path.exists('models'):
        os.mkdir('models')

    set_random_seed(args.seed, args)

    track = args.track
    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    prefix_05 = 'ASVspoof05'

    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr
    )
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = WrappedModel(model,device)
        model = model.to(device)
        print('Model loaded : {}'.format(args.model_path))

    if args.eval:
        eval_proto_path = os.path.join(
            args.protocols_path
            + ASVspoof_path_name
        )

        file_eval = genSpoof_list(
            dir_meta=eval_proto_path,
            is_train=False,
            is_eval=True,
        )
        print('no. of eval trials', len(file_eval))

        eval_set = Dataset_ASVspoof05_eval(
            list_IDs=file_eval,
            base_dir=os.path.join(args.database_path + folder_files),
            normalize = normalize)
        produce_evaluation_file(eval_set, model, device, args.eval_output, normalize)
        return

    train_proto_path = os.path.join(
        args.protocols_path
        + 'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    )
    d_label_trn, file_train = genSpoof_list(
        dir_meta=train_proto_path,
        is_train=True,
        is_eval=False,
    )
    print('no. of training trials', len(file_train))

    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_train/'),
        algo=args.algo,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    del train_set, d_label_trn

    dev_proto_path = os.path.join(
        args.protocols_path
        + 'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    )
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=dev_proto_path,
        is_train=False,
        is_eval=False,
    )
    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_dev/'),
        algo=args.algo,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
    )

    del dev_set, d_label_dev

    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    for epoch in range(num_epochs):
        running_loss = train_epoch(
            train_loader,
            model,
            args.lr,
            optimizer,
            device,
        )
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch, running_loss, val_loss))
        torch.save(
            model.state_dict(),
            os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)),
        )


if __name__ == '__main__':
    run_asvspoof2021_baseline()