# SingLEM transfer-learning pipeline (UI complexity)

Use the pretrained **single-channel** EEG foundation model
[SingLEM](https://github.com/ttlabtuat/SingLEM-EEG-Foundation-Model) as a frozen
feature extractor on our NeuroSky data, then train a small classifier head and
evaluate it **honestly** (LOSO + permutation test, and subject-specific
trial-grouped CV).

## What's here
| file | does |
|---|---|
| `config.py` | all paths + parameters (**edit the two paths at the top**) |
| `1_prepare_data.py` | `dataset_clean` → SingLEM format (onset-locked 2 s, 128 Hz, scaled to (−1,1)) |
| `2_extract_features.py` | frozen SingLEM encoder → 16-d feature per trial (uses GPU) |
| `3_classify.py` | RBF-SVM / Logistic head, LOSO + permutation test + subject-specific CV |

## Setup (one time)
```bash
# 1. clone the model + install
git clone https://github.com/ttlabtuat/SingLEM-EEG-Foundation-Model.git
cd SingLEM-EEG-Foundation-Model
pip install -r requirements.txt        # plus: pip install torch (CUDA build for your GPU)
# 2. make sure weights/singlem_pretrained.pt exists (ships with the repo)
```
Then edit **`config.py`**:
- `SINGLEM_REPO` → the cloned repo root.
- (binary is the best shot; for 3-class add `'design_C_moderate': 'Moderate'` to `LABEL_MAP` and `'Moderate'` to `CLASSES`.)

## Run order
```bash
cd Codes/singLEM
python 1_prepare_data.py        # CPU, ~1 min  -> prepared_data/*.pkl
python 2_extract_features.py    # GPU          -> singlem_features.npz
python 3_classify.py            # CPU, ~1 min  -> results/singlem_results.txt
```

## How to read the result
- **Baselines printed first:** chance (50% binary) and the majority baseline. Judge against these, not 0.
- **LOSO** = generalise to a *new person* (strict). **Subject-specific** = personalised (easier, legitimate).
- **Permutation p < 0.05** is the proof that any above-chance balanced accuracy is real, not noise.
- Realistic expectation given everything else (~0.53 ceiling): **~0.55–0.62 balanced**. SingLEM scored 82% on N-back (a cognitive-load task) on research-grade data, so there's a real chance it edges past 0.60 — but the NeuroSky dry-electrode **domain gap** may cap it. Either outcome is a clean, citable result.

## Knobs worth trying (in `config.py`)
- `SEG_SEC` — longer segment = more 1-s tokens (more context for the transformer) but drops short trials. Try 2 → 3.
- **Amplitude scaling** — we clip to ±100 then ÷100 to hit SingLEM's (−1,1) range (matches its ±100 µV artifact rule and our pipeline's µV labelling). If features look degenerate, try a robust per-recording scale instead (divide by ~3×std). This is the main domain-gap risk.
- **Fine-tuning** — we use the encoder *frozen* (fastest, what the paper did). If frozen LOSO clears chance, unfreezing the last transformer block + a head and fine-tuning per-fold may add a little (watch for overfitting; keep LOSO + permutation).

## Important
- `2_extract_features.py` is written against the README API (`from SingLEM.model import EEGEncoder, Config`). If the cloned repo's module layout differs, adjust the import inside `load_singlem_encoder()` — everything else is generic.
- Keep the validation honest: never switch LOSO/trial-grouped to a random window/trial split, or the SingLEM number inflates exactly like the 74% demo did.
