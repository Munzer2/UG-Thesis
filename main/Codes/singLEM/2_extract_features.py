"""
Step 2 — Run the FROZEN SingLEM encoder to turn each trial into a feature vector.

Pipeline per trial:
  fixed 256-sample segment  ->  tokenize into 1-s tokens (128 samples, stride 96)
  ->  SingLEM encoder (no grad)  ->  (num_tokens, 16)  ->  mean over tokens  ->  16-d feature

Output: FEAT_FILE (.npz) with X (n_trials, 16), y, subjects, trials.

NOTE ON THE IMPORT: this is written against the API shown in the SingLEM README
(`from SingLEM.model import EEGEncoder, Config`). If the cloned repo uses different
module names, adjust `load_singlem_encoder()` below — the rest is generic.
"""
import os, sys, glob, pickle
import numpy as np
import torch
import config as C


def tokenize(seg, token=C.TOKEN, stride=C.STRIDE):
    """(n_samples,) -> (num_tokens, token), sliding window; always include the tail token."""
    n = len(seg)
    starts = list(range(0, n - token + 1, stride))
    if not starts or starts[-1] != n - token:
        starts.append(n - token)
    return np.stack([seg[s:s + token] for s in starts]).astype(np.float32)


def load_singlem_encoder(device):
    # match the repo's examples: add the SingLEM package dir and `from model import ...`
    sys.path.insert(0, os.path.join(C.SINGLEM_REPO, 'SingLEM'))
    from model import EEGEncoder, Config
    cfg = Config()
    cfg.mask_prob = 0.0                                          # no masking at inference (required)
    enc = EEGEncoder(cfg)
    state = torch.load(C.SINGLEM_WEIGHTS, map_location=device)
    state = state.get('model', state) if isinstance(state, dict) and 'model' in state else state
    enc.load_state_dict(state)
    enc.eval().to(device)
    return enc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    enc = load_singlem_encoder(device)

    pkls = sorted(glob.glob(os.path.join(C.PREP_DIR, "*.pkl")))
    if not pkls:
        raise SystemExit(f"No prepared .pkl in {C.PREP_DIR}. Run 1_prepare_data.py first.")

    X, y, subjects, trials = [], [], [], []
    for pk in pkls:
        with open(pk, 'rb') as fh:
            d = pickle.load(fh)
        # tokenize every trial -> (n_trials, num_tokens, 128)
        toks = np.stack([tokenize(seg[0]) for seg in d['data']])   # (n_trials, T, 128)
        tens = torch.tensor(toks, dtype=torch.float32, device=device)
        T = tens.shape[1]
        feats = []
        with torch.no_grad():
            for i in range(0, len(tens), 64):                   # batch for the GPU
                xb = tens[i:i + 64]
                out = enc(xb)
                emb = out[0] if isinstance(out, (tuple, list)) else out   # (B*T, 16)  -- flattened!
                emb = emb.view(xb.shape[0], T, -1)              # -> (B, T, 16)
                if C.AGG == 'mean':
                    emb = emb.mean(dim=1)                        # (B, 16)
                else:                                           # 'concat' (paper): keep all tokens
                    emb = emb.reshape(xb.shape[0], -1)          # (B, T*16)
                feats.append(emb.cpu().numpy())
        feats = np.concatenate(feats, 0)
        X.append(feats); y.extend(d['labels']); subjects.extend([d['subject']] * len(feats))
        trials.extend(d['trials'])
        print(f"  {d['subject']:<28} {len(feats):>3} trials -> {feats.shape[1]}-d features")

    X = np.concatenate(X, 0)
    np.savez(C.FEAT_FILE, X=X, y=np.array(y), subjects=np.array(subjects), trials=np.array(trials))
    print(f"\nDONE: {X.shape[0]} trials x {X.shape[1]} features -> {C.FEAT_FILE}")


if __name__ == "__main__":
    main()
