import os
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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


def main():
    ###################################################################
    # כאן משנים את ההגדרות לפי מה שאתה צריך
    ###################################################################
    META_FILE = "/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/cm_protocols/ASVspoof5.train.metadata.txt"
    AUDIO_ROOT = "/gpfs0/bgu-benshimo/projects/ASVspoof5/flac_T/"
    OUTPUT_DIR = "/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/xlsr_feats/"

    MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
    AUDIO_EXT = ".flac"          # אם זה wav, תשנה ל-".wav"
    BATCH_SIZE = 10              # כמה utterances בריצה אחת של המודל
    CHUNK_SIZE = 100000          # כל כמה utterances נשמר קובץ npz
    DEVICE_NAME = "cuda"         # או "cpu"
    ###################################################################

    device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model and processor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )
    model = Wav2Vec2Model.from_pretrained(
        MODEL_NAME
    )
    model.to(device)
    model.eval()

    
    entries = load_metadata(META_FILE)
    print(f"Loaded {len(entries)} metadata lines")

    feats_chunk = []
    speakers_chunk = []
    utt_ids_chunk = []
    genders_chunk = []
    attacks_chunk = []
    labels_chunk = []
    chunk_idx = 0
    in_chunk_counter = 0

    raw_batch = []
    batch_meta = []

    pbar = tqdm(entries, desc="Processing utterances")
    for entry in pbar:
        utt_id = entry["utt_id"]
        audio_path = os.path.join(AUDIO_ROOT, utt_id + AUDIO_EXT)

        if not os.path.isfile(audio_path):
            print(f"Warning: audio file not found: {audio_path}")
            continue

        wav, _ = load_audio(audio_path, target_sr=16000)
        raw_batch.append(wav)
        batch_meta.append(entry)

        if len(raw_batch) == BATCH_SIZE:
            inputs = feature_extractor(
                raw_batch,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_values=input_values,
                                attention_mask=attention_mask)
                hidden = outputs.last_hidden_state    # [B, T, H]
                emb = hidden.mean(dim=1).cpu().numpy()  # [B, H]

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

            raw_batch = []
            batch_meta = []

    if len(raw_batch) > 0:
        inputs = feature_extractor(
            raw_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values,
                            attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            emb = hidden.mean(dim=1).cpu().numpy()

        for e, vec in zip(batch_meta, emb):
            feats_chunk.append(vec)
            speakers_chunk.append(e["speaker"])
            utt_ids_chunk.append(e["utt_id"])
            genders_chunk.append(e["gender"])
            attacks_chunk.append(e["attack"])
            labels_chunk.append(e["label"])
            in_chunk_counter += 1

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
