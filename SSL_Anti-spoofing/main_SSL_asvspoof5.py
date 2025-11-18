import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
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
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=14, shuffle=False, drop_last=False,num_workers=0)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in tqdm(data_loader, total=len(data_loader)):
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

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
        batch_out = model(batch_x)
        
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
            if self.normalize == True:
                x = x / x.abs().max()
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id  


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
    normalize = False

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
        produce_evaluation_file(eval_set, model, device, args.eval_output)
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