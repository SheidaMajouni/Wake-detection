"""
Train File-level Wake vs No-Wake LSTM Classifier
------------------------------------------------
- Positives: processed_ts/{train,valid,test}/*.csv  (columns: t_s, z_m, wake_label)
- Negatives: vessel_NO_wake_timeseries/{train,valid,test}/*.txt (JSON with z_positions_m)
- Each file is one sample; all sequences are resampled to TARGET_LEN.
- Saves: model + scaler + best threshold in a single .pt checkpoint.

Run:
  python lstm_train_filelevel.py \
    --pos_root processed_ts \
    --neg_root ../marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac/data/vessel_NO_wake_timeseries \ # downloaded from backgrouond
    --out runs-lstm-file \
    --epochs 30 --batch 64
"""

import os, glob, json, argparse, random
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------- Repro --------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Defaults --------
SAMPLES_PER_MIN = 300          # 200 ms step
TARGET_LEN = 3000              # 10 minutes
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 7

# -------- IO helpers --------
def scan(root, pattern="*.csv"):
    files = glob.glob(os.path.join(root, pattern))
    files.sort()
    return files

def load_pos_csv(fp):
    df = pd.read_csv(fp).sort_values("t_s").reset_index(drop=True)
    return df  # must contain t_s, z_m, wake_label

def load_neg_txt(fp):
    with open(fp, "r") as f:
        d = json.load(f)
    z = np.asarray(d["z_positions_m"], dtype=np.float32)
    return z

def resample_to_len(z, target_len=TARGET_LEN):
    if len(z) == target_len:
        return z.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, num=len(z), endpoint=True)
    x_tgt = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    z_tgt = np.interp(x_tgt, x_src, z)
    return z_tgt.astype(np.float32)

# -------- Dataset --------
class FileLevelDataset(Dataset):
    """
    Mix of positive CSVs (with labels) and negative TXTs (no labels in file).
    Scaling: z-score with a StandardScaler fitted on train set.
    """
    def __init__(self, pos_files, neg_files, fit_scaler=False, scaler=None):
        self.samples = []  # (z_scaled, y, src)
        raws = []

        # positives
        for fp in pos_files:
            df = load_pos_csv(fp)
            z = df["z_m"].to_numpy(dtype=np.float32)
            z = resample_to_len(z, TARGET_LEN)
            raws.append(z)
            self.samples.append((None, 1, fp))

        # negatives
        for fp in neg_files:
            try:
                z = load_neg_txt(fp)
                z = resample_to_len(z, TARGET_LEN)
                raws.append(z)
                self.samples.append((None, 0, fp))
            except Exception as e:
                print(f"[WARN] skip {fp}: {e}")

        raw = np.stack(raws, axis=0) if len(raws) else np.zeros((0, TARGET_LEN), dtype=np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(raw.reshape(-1,1)).reshape(raw.shape)
        else:
            if scaler is None:
                raise ValueError("Provide scaler when fit_scaler=False")
            self.scaler = scaler
            scaled = self.scaler.transform(raw.reshape(-1,1)).reshape(raw.shape)

        for i, (zs, (_, y, fp)) in enumerate(zip(scaled, self.samples)):
            self.samples[i] = (zs.astype(np.float32), y, fp)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        z, y, fp = self.samples[idx]
        # return shape (B,1,L)
        return torch.from_numpy(z[None, :]).float(), torch.tensor(float(y)).float(), fp

# -------- Model --------
class LSTMWakeFile(nn.Module):
    def __init__(self, input_size=1, hidden=128, num_layers=2, bidir=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidir, dropout=dropout if num_layers>1 else 0.0
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):           # x: (B,1,L)
        x = x.transpose(1, 2)       # (B,L,1)
        h, _ = self.lstm(x)         # (B,L,H*)
        hmax = h.max(dim=1).values  # temporal max-pool
        logit = self.head(hmax).squeeze(1)
        return logit

# -------- Metrics --------
def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"Accuracy":acc, "Precision":p, "Recall":r, "F1":f1, "TP":int(tp), "FP":int(fp),
            "TN":int(tn), "FN":int(fn)}

def eval_loader(model, loader, criterion, thr=0.5):
    model.eval()
    losses, y_true, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.append(prob); y_true.append(yb.cpu().numpy())
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    metrics = compute_metrics(y_true, y_prob, thr=thr)
    return float(np.mean(losses)), y_true, y_prob, metrics

def best_threshold(y_true, y_prob):
    ts = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1, 0, 0)
    for t in ts:
        m = compute_metrics(y_true, y_prob, thr=t)
        if m["F1"] > best[1]:
            best = (t, m["F1"], m["Precision"], m["Recall"])
    return best  # thr, F1, P, R

# -------- Main --------
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # Build file lists
    pos = {s: scan(os.path.join(args.pos_root, s), "*.csv") for s in ("train","valid","test")}
    neg = {s: scan(os.path.join(args.neg_root, s), "*.txt") for s in ("train","valid","test")}

    # Balance negatives by undersampling to #positives per split
    rng = np.random.default_rng(SEED)
    neg_bal = {}
    for s in ("train","valid","test"):
        pos_n = len(pos[s]); neg_all = neg[s]
        if len(neg_all) <= pos_n:
            neg_bal[s] = neg_all
        else:
            idx = rng.choice(len(neg_all), size=pos_n, replace=False)
            neg_bal[s] = [neg_all[i] for i in idx]

    # Datasets
    train_ds = FileLevelDataset(pos["train"], neg_bal["train"], fit_scaler=True)
    scaler = train_ds.scaler
    valid_ds = FileLevelDataset(pos["valid"], neg_bal["valid"], fit_scaler=False, scaler=scaler)
    test_ds  = FileLevelDataset(pos["test"],  neg_bal["test"],  fit_scaler=False, scaler=scaler)

    # Loaders
    def L(ds, sh): return DataLoader(ds, batch_size=args.batch, shuffle=sh, drop_last=False)
    train_loader = L(train_ds, True)
    valid_loader = L(valid_ds, False)
    test_loader  = L(test_ds,  False)

    # Stats
    for name, ds in [("train",train_ds),("valid",valid_ds),("test",test_ds)]:
        ys = [int(y.item()) for (_,y,_) in ds]
        c = Counter(ys)
        print(f"[STATS] {name}: N={len(ds)} | pos={c.get(1,0)} | neg={c.get(0,0)}")

    # Model / opt
    model = LSTMWakeFile().to(device)
    criterion = nn.BCEWithLogitsLoss()   # class balance handled by undersampling
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val, best_state, best_thr = -1.0, None, 0.5
    no_improve = 0

    for ep in range(1, args.epochs+1):
        model.train()
        tr_losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses))

        # Validate (sweep threshold on val)
        val_loss, yv_true, yv_prob, _ = eval_loader(model, valid_loader, criterion, thr=0.5)
        thr, f1b, pb, rb = best_threshold(yv_true, yv_prob)
        scheduler.step(f1b)

        print(f"Epoch {ep:03d} | TrainLoss {tr_loss:.4f} | ValLoss {val_loss:.4f} | "
              f"Val(F1/P/R@thr={thr:.2f}) {f1b:.3f}/{pb:.3f}/{rb:.3f}")

        if f1b > best_val + 1e-6:
            best_val, best_state, best_thr = f1b, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, thr
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.es_patience:
                print(f"[STOP] early stop at epoch {ep} (best Val F1={best_val:.3f})")
                break

    if best_state is not None:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

    # Final metrics with best threshold
    tr_loss, ytr_true, ytr_prob, tr_m = eval_loader(model, train_loader, criterion, thr=best_thr)
    va_loss, yva_true, yva_prob, va_m = eval_loader(model, valid_loader, criterion, thr=best_thr)
    te_loss, yte_true, yte_prob, te_m = eval_loader(model, test_loader,  criterion, thr=best_thr)

    print("\n===== FINAL (file-level) =====")
    print(f"[train] thr={best_thr:.2f} loss={tr_loss:.4f} -> {tr_m}")
    print(f"[valid] thr={best_thr:.2f} loss={va_loss:.4f} -> {va_m}")
    print(f"[test ] thr={best_thr:.2f} loss={te_loss:.4f} -> {te_m}")

    # Save model + scaler + threshold
    os.makedirs(args.out, exist_ok=True)
    ckpt = {
        "model_state": {k:v.cpu() for k,v in model.state_dict().items()},
        "scaler_mean": train_ds.scaler.mean_.astype(np.float32),
        "scaler_scale": train_ds.scaler.scale_.astype(np.float32),
        "target_len": TARGET_LEN,
        "threshold": float(best_thr),
        "meta": {"epochs": args.epochs, "batch": args.batch}
    }
    save_path = os.path.join(args.out, f"lstm_file_best.pt")
    torch.save(ckpt, save_path)
    print(f"[SAVE] {save_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_root", type=str, required=True)
    ap.add_argument("--neg_root", type=str, required=True)
    ap.add_argument("--out",      type=str, default="runs-lstm-file")
    ap.add_argument("--epochs",   type=int, default=EPOCHS)
    ap.add_argument("--batch",    type=int, default=BATCH_SIZE)
    ap.add_argument("--lr",       type=float, default=LR)
    ap.add_argument("--wd",       type=float, default=WEIGHT_DECAY)
    ap.add_argument("--es_patience", type=int, default=EARLY_STOP_PATIENCE)
    args = ap.parse_args()
    main(args)
