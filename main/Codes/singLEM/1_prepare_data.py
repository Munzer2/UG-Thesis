"""
Step 1 — Convert dataset_clean into SingLEM's expected format.

For each trial we take a FIXED, onset-locked segment (first SEG_SEC seconds after
image onset), so the model can't use trial duration (the reaction-time confound),
then preprocess to SingLEM's spec:
    clip to +/-100  ->  divide by 100  (amplitudes in (-1,1))  ->  resample 512->128 Hz

Output: one .pkl per subject in PREP_DIR with
    {'data': (n_trials, 1, SEG_SAMPLES_OUT), 'labels': [...], 'trials': [...], 'subject': name}
This matches SingLEM's downstream format (num_trials, num_channels, num_samples).
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import config as C


def prepare():
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, "UI_Exp_*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {C.DATA_DIR}")
    print(f"Preparing {len(files)} sessions -> SingLEM format "
          f"({C.SEG_SEC}s onset-locked, {C.SR_OUT}Hz, {C.SEG_SAMPLES_OUT} samples/trial)\n")

    per_subject = {}
    total, dropped_short, dropped_artifact = 0, 0, 0
    for fp in files:
        df = pd.read_csv(fp)
        name = os.path.basename(fp).replace('UI_Exp_', '').replace('.csv', '').rsplit('_', 1)[0]
        name = C.PARTICIPANT_MERGE.get(name, name)
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        task = df[mask].copy()
        if task.empty:
            continue
        task['complexity'] = task['label'].map(C.LABEL_MAP)
        has_art = 'is_artifact' in task.columns

        for cls in C.CLASSES:
            comp = task[task['complexity'] == cls]
            for img, g in comp.groupby('image'):                  # one trial = one image
                v = g['value'].values.astype(np.float64)
                art = g['is_artifact'].values if has_art else np.zeros(len(v), bool)
                if len(v) < C.SEG_SAMPLES_IN:                     # too short for onset-locked segment
                    dropped_short += 1
                    continue
                seg = v[:C.SEG_SAMPLES_IN]                        # onset-locked, fixed length
                if has_art and art[:C.SEG_SAMPLES_IN].mean() > C.ARTIFACT_MAX_FRAC:
                    dropped_artifact += 1
                    continue
                seg = seg * C.RAW_TO_UV                                    # NeuroSky raw -> microvolts (~x0.22)
                seg = np.clip(seg, -C.AMP_CLIP_UV, C.AMP_CLIP_UV) / C.AMP_CLIP_UV  # uV -> (-1,1), SingLEM spec
                seg = resample_poly(seg, C.SR_OUT, C.SR_IN)        # 512 -> 128 Hz
                seg = seg[:C.SEG_SAMPLES_OUT]
                if len(seg) < C.SEG_SAMPLES_OUT:
                    seg = np.pad(seg, (0, C.SEG_SAMPLES_OUT - len(seg)))
                per_subject.setdefault(name, {'data': [], 'labels': [], 'trials': []})
                per_subject[name]['data'].append(seg.astype(np.float32))
                per_subject[name]['labels'].append(cls)
                per_subject[name]['trials'].append(f"{name}|{cls}|{img}")
                total += 1

    n_saved = 0
    for name, d in per_subject.items():
        data = np.stack(d['data'])[:, None, :]                    # (n_trials, 1, SEG_SAMPLES_OUT)
        out = {'data': data, 'labels': np.array(d['labels']),
               'trials': np.array(d['trials']), 'subject': name}
        with open(os.path.join(C.PREP_DIR, f"{name}.pkl"), 'wb') as fh:
            pickle.dump(out, fh)
        n_saved += 1
        cls_counts = {c: int((out['labels'] == c).sum()) for c in C.CLASSES}
        print(f"  {name:<28} {len(data):>3} trials  {cls_counts}")

    print(f"\nDONE: {total} trials from {n_saved} subjects -> {C.PREP_DIR}")
    print(f"  dropped (shorter than {C.SEG_SEC}s): {dropped_short}")
    print(f"  dropped (artifact-heavy):          {dropped_artifact}")
    print(f"  each trial: shape (1, {C.SEG_SAMPLES_OUT}) @ {C.SR_OUT} Hz, amplitudes in (-1,1)")


if __name__ == "__main__":
    prepare()
