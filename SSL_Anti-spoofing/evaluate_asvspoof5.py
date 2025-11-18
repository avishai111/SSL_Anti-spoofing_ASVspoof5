import sys
import os
import numpy as np
import pandas as pd
import eval_metrics_DF as em
from glob import glob


def eval_to_score_file(submit_file: str, truth_dir: str, phase: str):
    """
    מחשב EER עבור קובץ ציון נתון ושלב נתון (progress / eval / hidden_track)
    """

    cm_key_file = os.path.join(truth_dir, 'CM', 'trial_metadata.txt')

    # קורא את קובץ המטא-דטה ואת קובץ הציונים
    cm_data = pd.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pd.read_csv(
        submit_file, sep=' ', header=None, skipinitialspace=True
    )

    # מסנן רק את השורה של ה-phase הרלוונטי
    cm_data_phase = cm_data[cm_data[7] == phase]

    # בדיקת התאמת כמות שורות (רק עבור ה-phase הרלוונטי)
    if len(submission_scores) != len(cm_data_phase):
        print(
            'CHECK: submission has %d of %d expected trials for phase %s.'
            % (len(submission_scores), len(cm_data_phase), phase)
        )
        sys.exit(1)

    # מיזוג בין הציונים לבין המטא-דטה של ה-phase
    cm_scores = submission_scores.merge(
        cm_data_phase, left_on=0, right_on=1, how='inner'
    )

    # בדיקה בסיסית: כמה bona / spoof יש אחרי merge
    print("DEBUG: #rows after merge:", len(cm_scores))
    print("DEBUG: #bonafide:", (cm_scores[5] == 'bonafide').sum())
    print("DEBUG: #spoof   :", (cm_scores[5] == 'spoof').sum())

    # חילוץ ציוני bona/spoof
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    # חישוב EER
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100 * eer_cm)
    print(out_data)
    return eer_cm


def main():
    # כאן אפשר להשתמש ב־sys.argv כמו במקור
    if len(sys.argv) != 4:
        print("CHECK: invalid input arguments.")
        print("Usage: python eval_cm.py <submit_file> <truth_dir> <phase>")
        sys.exit(1)

    # submit_file = sys.argv[1]
    # truth_dir = sys.argv[2]
    # phase = sys.argv[3]
    submit_file = "/path/to/your/submission.txt"
    truth_dir = "/path/to/truth_dir"
    phase = "eval"  # או "eval" / "hidden_track"

    eer = eval_to_score_file(submit_file, truth_dir, phase)

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % submit_file)
        sys.exit(1)

    if not os.path.isdir(truth_dir):
        print("%s doesn't exist" % truth_dir)
        sys.exit(1)

    if phase not in {"progress", "eval", "hidden_track"}:
        print("phase must be either progress, eval, or hidden_track")
        sys.exit(1)

    _ = eval_to_score_file(submit_file, truth_dir, phase)


if __name__ == "__main__":
    main()
