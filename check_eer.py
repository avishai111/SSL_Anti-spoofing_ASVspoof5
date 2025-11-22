import numpy as np
from sklearn.metrics import roc_curve
import os

def load_scores(path):
    trial_ids = []
    scores = []
    with open(path, "r") as f:
        for line in f:
            tid, sc = line.strip().split()
            trial_ids.append(tid)
            scores.append(float(sc))
    return np.array(trial_ids), np.array(scores)


def load_metadata(metadata_path):
    mapping = {}
    with open(metadata_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Metadata format:
            # speaker  trial_id   gender  -  attack  label
            # Example:
            # T_4850   T_0000000000   F   -   A05   spoof
            trial_id = parts[1]
            label = parts[-1]

            if label == "bonafide":
                mapping[trial_id] = 0
            else:
                mapping[trial_id] = 1
    return mapping


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


def evaluate(score_path, metadata_path):
    trial_ids, scores = load_scores(score_path)
    meta = load_metadata(metadata_path)

    labels = []
    for tid in trial_ids:
        if tid not in meta:
            raise ValueError(f"Missing label for trial {tid}")
        labels.append(meta[tid])
    
    labels = np.array(labels)
    eer = compute_eer(labels, scores)
    return eer


if __name__ == "__main__":
    metadata_path = "/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/cm_protocols/ASVspoof5.train.metadata.txt"

    score_files = {
        "no_normalize": "/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/train_asvspoof5_no_normalize.txt",
        "normalize": "/gpfs0/bgu-benshimo/users/wavishay/cm_analysis/train_asvspoof5_normalize.txt",
    }

    for name, path in score_files.items():
        print(f"Evaluating {name} ...")
        eer = evaluate(path, metadata_path)
        print(f"EER ({name}): {eer * 100:.4f}%")
        print()
