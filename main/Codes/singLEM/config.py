"""
Shared configuration for the SingLEM transfer-learning pipeline.
EDIT THE TWO PATHS AT THE TOP after you clone the SingLEM repo + download weights.
"""
import os

# ============================================================
# 1) SingLEM model — VENDORED in ./_repo (cloned from github.com/ttlabtuat/SingLEM, MIT).
#    Just copy this whole folder to the GPU machine; no re-clone needed.
#    (To use a clone elsewhere, point SINGLEM_REPO at that repo root.)
# ============================================================
SINGLEM_REPO    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_repo")
SINGLEM_WEIGHTS = os.path.join(SINGLEM_REPO, "weights", "singlem_pretrained.pt")

# ============================================================
# 2) Your data (already preprocessed) — relative to this folder
# ============================================================
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(HERE, "..", "..", "dataset_clean")     # UI_Exp_*.csv
PREP_DIR    = os.path.join(HERE, "prepared_data")                 # per-subject .pkl written here
FEAT_FILE   = os.path.join(HERE, "singlem_features.npz")          # embeddings written here
RESULTS_DIR = os.path.join(HERE, "results")

# ============================================================
# 3) Task / preprocessing-to-SingLEM-spec parameters
# ============================================================
# Classes: binary is the best shot. For 3-class add 'design_C_moderate'->'Moderate'.
LABEL_MAP = {'design_A_simple': 'Simple', 'design_B_complex': 'Complex'}
CLASSES   = ['Simple', 'Complex']

PARTICIPANT_MERGE = {'adnan2': 'adnan', 'Mushfiq2': 'Mushfiq'}

# SingLEM input spec (from the paper/README)
SR_IN   = 512        # your raw sampling rate
SR_OUT  = 128        # SingLEM expects 128 Hz
TOKEN   = 128        # 1-second token = 128 samples @128 Hz
STRIDE  = 96         # 250 ms overlap (stride 96) -- matches SingLEM demo (overlap=0.25s)
# dataset_clean is ALREADY in microvolts (preprocess.py now converts raw*0.2197 -> uV up front),
# so NO further conversion here (set to 1.0 to avoid double-converting).
RAW_TO_UV   = 1.0
AMP_CLIP_UV = 100.0  # SingLEM expects uV; clip to +/-100 uV then /100 -> (-1,1) (its artifact rule + scaling)
AGG = 'concat'       # how to combine per-token features into a trial vector: 'concat' (paper) or 'mean'

# Onset-locked fixed segment per trial (avoids the trial-DURATION confound).
# 2 s keeps ~80-90% of trials; raise for more tokens/context (drops short trials).
SEG_SEC = 2.0
SEG_SAMPLES_IN  = int(SEG_SEC * SR_IN)    # 1024 @512 Hz
SEG_SAMPLES_OUT = int(SEG_SEC * SR_OUT)   # 256  @128 Hz
ARTIFACT_MAX_FRAC = 0.4                    # drop a trial if >40% of its segment was flagged artifact

SEED = 42

os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
