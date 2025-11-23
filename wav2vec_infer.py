import os
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import warnings
warnings.filterwarnings("ignore", message=".*torchaudio.load.*")
import numpy as np

def normalize_peak(wav: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Peak normalization: מחלקים במקסימום המוחלט כך שהאמפליטודה תהיה בטווח [-1, 1].
    """
    # לוודא float32
    wav = wav.astype(np.float32)

    peak = np.max(np.abs(wav))

    return wav / peak





def load_metadata(meta_path):
    """
    קורא קובץ מטה־דאטה בפורמט:
    speaker utt_id gender - attack label
    """
    entries = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            speaker = parts[0]
            utt_id = parts[1]
            gender = parts[2]
            attack = parts[4]
            label = parts[5]

            entries.append(
                dict(
                    speaker=speaker,
                    utt_id=utt_id,
                    gender=gender,
                    attack=attack,
                    label=label,
                )
            )
    return entries


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)  # [channels, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # המרת סטריאו למונו
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    wav = wav.squeeze(0).cpu().numpy().astype("float32")
    return wav, target_sr  # [time]


def save_chunk(chunk_idx, feats, speakers, utt_ids, genders, attacks, labels, out_dir):
    if len(feats) == 0:
        return

    feats_arr = np.stack(feats, axis=0).astype(np.float32)
    speakers_arr = np.array(speakers)
    utt_ids_arr = np.array(utt_ids)
    genders_arr = np.array(genders)
    attacks_arr = np.array(attacks)
    labels_arr = np.array(labels)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"xlsr_feats_part{chunk_idx:03d}.npz")
    np.savez(
        out_path,
        feats=feats_arr,
        speaker=speakers_arr,
        utt_id=utt_ids_arr,
        gender=genders_arr,
        attack=attacks_arr,
        label=labels_arr,
    )
    print(f"Saved {len(feats)} utterances to {out_path}")

import os
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def load_metadata(meta_path):
    """
    קובץ בפורמט:
    speaker utt_id gender - attack label
    """
    entries = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            speaker = parts[0]
            utt_id = parts[1]
            gender = parts[2]
            attack = parts[4]
            label = parts[5]

            entries.append(
                dict(
                    speaker=speaker,
                    utt_id=utt_id,
                    gender=gender,
                    attack=attack,
                    label=label,
                )
            )
    return entries


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)  # [channels, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # סטריאו → מונו
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    # מחזירים numpy 1D
    wav = wav.squeeze(0).cpu().numpy().astype("float32")
    return wav, target_sr


def save_chunk(chunk_idx, feats, speakers, utt_ids, genders, attacks, labels, out_dir):
    if len(feats) == 0:
        return

    feats_arr = np.stack(feats, axis=0).astype(np.float32)
    speakers_arr = np.array(speakers)
    utt_ids_arr = np.array(utt_ids)
    genders_arr = np.array(genders)
    attacks_arr = np.array(attacks)
    labels_arr = np.array(labels)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"xlsr_feats_part{chunk_idx:03d}.npz")
    np.savez(
        out_path,
        feats=feats_arr,
        speaker=speakers_arr,
        utt_id=utt_ids_arr,
        gender=genders_arr,
        attack=attacks_arr,
        label=labels_arr,
    )
    print(f"Saved {len(feats)} utterances to {out_path}")


class AsvspoofAudioDataset(Dataset):
    def __init__(self, entries, audio_root, audio_ext=".flac", target_sr=16000,normalize = True):
        self.entries = entries
        self.audio_root = audio_root
        self.audio_ext = audio_ext
        self.target_sr = target_sr
        self.normalize = normalize
        
        print("normalize: ",self.normalize)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        utt_id = entry["utt_id"]
        audio_path = os.path.join(self.audio_root, utt_id + self.audio_ext)

        if not os.path.isfile(audio_path):
            # אתה יכול לשנות כאן להתנהגות אחרת אם חסר קובץ
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav, _ = load_audio(audio_path, target_sr=self.target_sr)
        if self.normalize:
          wav = normalize_peak(wav)
        return wav, entry


def collate_fn(batch):
    """
    batch הוא רשימה של (wav, entry).
    מחזירים:
      wavs: list[np.ndarray 1D]
      metas: list[dict]
    """
    wavs = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    return wavs, metas


def main():
    ###################################################################
    # הגדרות
    ###################################################################
    META_FILE = "/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/cm_protocols/ASVspoof5.train.metadata.txt"
    AUDIO_ROOT = "/gpfs0/bgu-benshimo/projects/ASVspoof5/flac_T/"
    OUTPUT_DIR = "/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/xlsr_feats/normalize/"
    normalize = True
    MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
    AUDIO_EXT = ".flac"
    BATCH_SIZE = 10
    CHUNK_SIZE = 100000      # כמה אוטרנסים לכל npz
    DEVICE_NAME = "cuda"
    TARGET_SR = 16000
    NUM_WORKERS = 4
    ###################################################################

    device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading feature extractor and model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    entries = load_metadata(META_FILE)
    print(f"Loaded {len(entries)} metadata lines")

    dataset = AsvspoofAudioDataset(
        entries=entries,
        audio_root=AUDIO_ROOT,
        audio_ext=AUDIO_EXT,
        target_sr=TARGET_SR,
        normalize = normalize,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    feats_chunk = []
    speakers_chunk = []
    utt_ids_chunk = []
    genders_chunk = []
    attacks_chunk = []
    labels_chunk = []
    chunk_idx = 0
    in_chunk_counter = 0

    pbar = tqdm(data_loader, desc="Processing utterances")
    for batch_wavs, batch_meta in pbar:
        # batch_wavs: list of np.ndarray 1D, אורכים שונים
        inputs = feature_extractor(
            batch_wavs,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values,
                            attention_mask=attention_mask)
            hidden = outputs.last_hidden_state        # [B, T, H]
            emb = hidden.mean(dim=1).cpu().numpy()    # [B, H]

        for e, vec in zip(batch_meta, emb):
            feats_chunk.append(vec)
            speakers_chunk.append(e["speaker"])
            utt_ids_chunk.append(e["utt_id"])
            genders_chunk.append(e["gender"])
            attacks_chunk.append(e["attack"])
            labels_chunk.append(e["label"])

            in_chunk_counter += 1
            if in_chunk_counter >= CHUNK_SIZE:
                save_chunk(
                    chunk_idx,
                    feats_chunk,
                    speakers_chunk,
                    utt_ids_chunk,
                    genders_chunk,
                    attacks_chunk,
                    labels_chunk,
                    OUTPUT_DIR,
                )
                chunk_idx += 1
                feats_chunk = []
                speakers_chunk = []
                utt_ids_chunk = []
                genders_chunk = []
                attacks_chunk = []
                labels_chunk = []
                in_chunk_counter = 0

    if in_chunk_counter > 0:
        save_chunk(
            chunk_idx,
            feats_chunk,
            speakers_chunk,
            utt_ids_chunk,
            genders_chunk,
            attacks_chunk,
            labels_chunk,
            OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
