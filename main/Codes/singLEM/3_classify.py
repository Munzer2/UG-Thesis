"""
Step 3 — Honest evaluation of the SingLEM features.

Trains a small head on the 16-d SingLEM embeddings and evaluates two ways, the
SAME honest protocols we used everywhere else:
  * LOSO (leave-one-subject-out)        -> generalise to a NEW person
  * Subject-specific, trial-grouped CV  -> personalised, no within-subject leak
Both report balanced accuracy + F1-macro vs the majority baseline, and LOSO gets
a PERMUTATION TEST (shuffle labels, redo CV) so any above-chance result is proven.

Classifiers: RBF-SVM (what the SingLEM paper used) + Logistic Regression.
"""
import os
import numpy as np
import config as C
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

CLFS = {
    'SVM (RBF)':     lambda: make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced')),
    'Logistic Reg':  lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
}
N_PERM = 200


def loso(X, y, groups, make):
    logo = LeaveOneGroupOut()
    yt, yp = [], []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make().fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
    yt, yp = np.array(yt), np.array(yp)
    return balanced_accuracy_score(yt, yp), f1_score(yt, yp, average='macro', labels=C.CLASSES)


def subject_specific(X, y, subjects, trials, make):
    accs, f1s = [], []
    for s in np.unique(subjects):
        m = subjects == s
        Xs, ys, ts = X[m], y[m], trials[m]
        n = min((np.unique(ts[ys == c]).size for c in C.CLASSES), default=0)
        if n < 2:
            continue
        ns = max(2, min(5, n))
        sgkf = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=C.SEED)
        yt, yp = [], []
        for tr, te in sgkf.split(Xs, ys, groups=ts):
            if len(np.unique(ys[tr])) < 2:
                continue
            mdl = make().fit(Xs[tr], ys[tr]); yt.extend(ys[te]); yp.extend(mdl.predict(Xs[te]))
        if yt:
            accs.append(balanced_accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, average='macro', labels=C.CLASSES, zero_division=0))
    return np.mean(accs), np.mean(f1s), len(accs)


def main():
    d = np.load(C.FEAT_FILE, allow_pickle=True)
    X, y, subjects, trials = d['X'], d['y'], d['subjects'], d['trials']
    chance = 1.0 / len(C.CLASSES)
    maj = max((y == c).mean() for c in C.CLASSES)
    print(f"{len(X)} trials, {X.shape[1]} features, {np.unique(subjects).size} subjects")
    print(f"chance = {chance:.1%} | majority baseline = {maj:.1%}\n")

    lines = [f"SingLEM features — {' vs '.join(C.CLASSES)} | chance {chance:.1%}, majority {maj:.1%}", ""]
    rng = np.random.default_rng(C.SEED)
    for name, make in CLFS.items():
        ba, f1 = loso(X, y, subjects, make)
        ss_ba, ss_f1, nsub = subject_specific(X, y, subjects, trials, make)
        # permutation test on the LOSO balanced accuracy
        null = np.array([loso(X, rng.permutation(y), subjects, make)[0] for _ in range(N_PERM)])
        p = (np.sum(null >= ba) + 1) / (N_PERM + 1)
        print(f"{name}")
        print(f"  LOSO            : balanced acc {ba:.3f} | F1 {f1:.3f} | perm p={p:.3f} "
              f"({'ABOVE chance' if p < 0.05 else 'ns'}; null {null.mean():.3f})")
        print(f"  Subject-specific: balanced acc {ss_ba:.3f} | F1 {ss_f1:.3f}  (n={nsub} subjects)\n")
        lines += [f"{name}",
                  f"  LOSO  balanced_acc={ba:.3f}  F1={f1:.3f}  perm_p={p:.3f}  null={null.mean():.3f}",
                  f"  SubjSpecific balanced_acc={ss_ba:.3f}  F1={ss_f1:.3f}  (n={nsub})", ""]

    with open(os.path.join(C.RESULTS_DIR, 'singlem_results.txt'), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"-> saved {C.RESULTS_DIR}/singlem_results.txt")


if __name__ == "__main__":
    main()
