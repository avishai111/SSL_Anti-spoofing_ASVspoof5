import numpy as np
import os

# ================= GLOBAL SETTINGS =================
BASE_DIR = "/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/xlsr_feats/normalize/"

FILE_1 = "xlsr_feats_part000.npz"
FILE_2 = "xlsr_feats_part001.npz"

OUTPUT_FILE = "combined__mean_norm_wav2vec.npz"

# איזה מפתחות 1D נרצה לשמר ולחבר
META_KEYS = ["utt_id", "speaker", "gender", "attack", "label"]
# ====================================================

files = [
    os.path.join(BASE_DIR, FILE_1),
    os.path.join(BASE_DIR, FILE_2),
]

all_feats = []
meta_lists = {k: [] for k in META_KEYS}

for f in files:
    print(f"Loading: {f}")
    data = np.load(f)

    # feats
    if "feats" not in data.files:
        raise ValueError(f"'feats' key not found in {f}")
    feats_arr = data["feats"]
    if feats_arr.ndim != 2:
        raise ValueError(f"'feats' in {f} is not 2D, shape={feats_arr.shape}")

    print(f"  feats shape: {feats_arr.shape}")
    all_feats.append(feats_arr)

    # מטה־דאטה 1D לפי שמות מוגדרים
    for key in META_KEYS:
        if key in data.files:
            arr = data[key]
            if arr.ndim != 1:
                raise ValueError(f"'{key}' in {f} is not 1D, shape={arr.shape}")
            print(f"  {key} shape: {arr.shape}")
            meta_lists[key].append(arr)
        else:
            print(f"  WARNING: key '{key}' not found in {f}")

# חיבור כל הפיצ'רים
combined_feats = np.concatenate(all_feats, axis=0)
print("Combined feats shape:", combined_feats.shape)

# חיבור כל המטה־דאטה לפי key
combined_meta = {}
for key, lst in meta_lists.items():
    combined_meta[key] = np.concatenate(lst, axis=0)
    print(f"Combined {key} shape:", combined_meta[key].shape)

output_path = os.path.join(BASE_DIR, OUTPUT_FILE)

# שמירה לקובץ npz אחד
np.savez(
    output_path,
    feats=combined_feats,
    **combined_meta,   # speaker, utt_id, gender, attack, label
)

print("Saved merged file to:", output_path)
