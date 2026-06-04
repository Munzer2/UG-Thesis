"""
Step 4 (optional, GPU) — FINE-TUNE SingLEM on our data.

Unlike step 2/3 (frozen encoder + linear probe), this trains the encoder itself:
  SingLEM encoder  ->  mean-pool tokens  ->  small MLP head  ->  Simple/Complex
By default it unfreezes only the LAST transformer block (+ ln_final + DimRedLayer)
and the head, keeping the rest frozen — the safe choice for small data (~445 trials).

Validation stays honest: leave-one-subject-out (a fresh model per fold, started
from the pretrained weights), reported vs chance + majority baseline. An optional
permutation test is available but EXPENSIVE (retrains every fold per shuffle).

Run on GPU:   python 4_finetune.py
Quick check:  python 4_finetune.py smoke      (2 folds, 2 epochs — just verifies it runs)

Honest expectation: ~0.54-0.58 balanced. Overfitting (445 trials) fights the gain;
a jump past 0.60 is unlikely given everything converges at ~0.53.
"""
import os, sys, glob, pickle
import numpy as np
import torch
import torch.nn as nn
import config as C
from sklearn.metrics import balanced_accuracy_score, f1_score

# ---------------- fine-tuning hyperparameters ----------------
UNFREEZE   = 'last_block'   # 'head' | 'last_block' | 'all'
FT_EPOCHS  = 25
FT_BATCH   = 32
LR_HEAD    = 1e-3
LR_ENC     = 1e-4           # small lr for pretrained weights
WEIGHT_DECAY = 1e-3
DROPOUT    = 0.5
N_PERM_FT  = 0             # 0 = skip permutation (set e.g. 20 on GPU if you must; very slow)
SEED       = C.SEED
torch.manual_seed(SEED); np.random.seed(SEED)

sys.path.insert(0, os.path.join(C.SINGLEM_REPO, 'SingLEM'))
from model import EEGEncoder, Config

CLS2I = {c: i for i, c in enumerate(C.CLASSES)}


def tokenize(seg, token=C.TOKEN, stride=C.STRIDE):
    n = len(seg); starts = list(range(0, n - token + 1, stride))
    if not starts or starts[-1] != n - token:
        starts.append(n - token)
    return np.stack([seg[s:s + token] for s in starts]).astype(np.float32)


def load_tokens():
    X, y, subj = [], [], []
    for pk in sorted(glob.glob(os.path.join(C.PREP_DIR, "*.pkl"))):
        d = pickle.load(open(pk, 'rb'))
        for seg, lab in zip(d['data'], d['labels']):
            X.append(tokenize(seg[0])); y.append(CLS2I[lab]); subj.append(d['subject'])
    return np.stack(X), np.array(y), np.array(subj)   # X: (N, T, 128)


class SingLEMClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        r = encoder.config.rep_dim
        self.head = nn.Sequential(nn.LayerNorm(r), nn.Linear(r, 64), nn.ELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(64, n_classes))

    def forward(self, x):                     # x: (B, T, 128)
        B, T = x.shape[0], x.shape[1]
        reps, _, _ = self.encoder(x)          # (B*T, rep_dim)
        reps = reps.view(B, T, -1).mean(dim=1)  # mean-pool tokens -> (B, rep_dim)
        return self.head(reps)


def set_trainable(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    if UNFREEZE in ('last_block', 'all'):
        keys = ['DimRedLayer', 'TransformerEncoder.ln_final']
        if UNFREEZE == 'last_block':
            keys.append('TransformerEncoder.Transformer_layers.11')
        for n, p in model.encoder.named_parameters():
            if UNFREEZE == 'all' or any(k in n for k in keys):
                p.requires_grad = True
    # head is always trainable
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(device):
    cfg = Config(); cfg.mask_prob = 0.0
    enc = EEGEncoder(cfg)
    enc.load_state_dict(torch.load(C.SINGLEM_WEIGHTS, map_location=device))
    m = SingLEMClassifier(enc, len(C.CLASSES)).to(device)
    set_trainable(m)
    return m


def train_fold(model, Xtr, ytr, device, epochs):
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder')]
    head_p = [p for p in model.head.parameters()]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': LR_ENC},
                             {'params': head_p, 'lr': LR_HEAD}], weight_decay=WEIGHT_DECAY)
    w = torch.tensor([len(ytr) / (len(C.CLASSES) * max(1, (ytr == k).sum())) for k in range(len(C.CLASSES))],
                     dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), FT_BATCH):
            idx = perm[i:i + FT_BATCH]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()


@torch.no_grad()
def predict(model, Xte, device):
    model.eval()
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xte), 128):
        out.append(model(Xte[i:i + 128]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def loso(X, y, subjects, device, epochs, folds=None):
    subs = np.unique(subjects)
    if folds:
        subs = subs[:folds]
    yt, yp, per = [], [], []
    for s in subs:
        tr, te = subjects != s, subjects == s
        if te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = build_model(device)
        train_fold(m, X[tr], y[tr], device, epochs)
        p = predict(m, X[te], device)
        yt.extend(y[te]); yp.extend(p)
        per.append(balanced_accuracy_score(y[te], p))
    yt, yp = np.array(yt), np.array(yp)
    return balanced_accuracy_score(yt, yp), f1_score(yt, yp, average='macro'), np.mean(per), np.std(per)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, y, subjects = load_tokens()
    chance = 1.0 / len(C.CLASSES); maj = max((y == k).mean() for k in range(len(C.CLASSES)))
    print(f"Device: {device} | {len(X)} trials, tokens/trial={X.shape[1]}, {np.unique(subjects).size} subjects")
    print(f"Fine-tune mode: {UNFREEZE} | chance {chance:.1%}, majority {maj:.1%}")
    print(f"Trainable params: {set_trainable(build_model(device)):,}\n")

    if smoke:
        ba, f1, m_, s_ = loso(X, y, subjects, device, epochs=2, folds=2)
        print(f"[SMOKE] 2 folds x 2 epochs -> balanced acc {ba:.3f} (just checks it runs)")
        return

    ba, f1, mfold, sfold = loso(X, y, subjects, device, FT_EPOCHS)
    print(f"FINE-TUNED LOSO: balanced acc {ba:.3f} | F1 {f1:.3f} | per-fold {mfold:.3f}±{sfold:.3f}")
    print(f"  (chance {chance:.1%}, majority {maj:.1%}; frozen linear-probe was ~0.54)")

    line = f"SingLEM FINE-TUNED ({UNFREEZE}) LOSO: balanced_acc={ba:.3f} F1={f1:.3f} perfold={mfold:.3f}+/-{sfold:.3f} (chance {chance:.2f})"
    if N_PERM_FT > 0:
        print(f"\nPermutation test ({N_PERM_FT} shuffles — expensive)...")
        rng = np.random.default_rng(SEED)
        null = []
        for i in range(N_PERM_FT):
            ba_n, *_ = loso(X, rng.permutation(y), subjects, device, FT_EPOCHS)
            null.append(ba_n); print(f"  {i+1}/{N_PERM_FT} null={np.mean(null):.3f}")
        p = (np.sum(np.array(null) >= ba) + 1) / (N_PERM_FT + 1)
        line += f" | perm_p={p:.3f} null={np.mean(null):.3f}"
        print(f"  permutation p = {p:.3f} ({'ABOVE chance' if p < 0.05 else 'ns'})")

    with open(os.path.join(C.RESULTS_DIR, 'singlem_finetune_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write(line + "\n")
    print(f"-> saved {C.RESULTS_DIR}/singlem_finetune_results.txt")


if __name__ == "__main__":
    main()
