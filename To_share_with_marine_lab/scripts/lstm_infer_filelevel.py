"""
Inference for File-level LSTM Classifier
----------------------------------------
- Loads a .pt checkpoint saved by train_lstm_filelevel.py (includes scaler + best threshold).
- Runs on a folder of files:
    - CSV (positives format): columns t_s, z_m, wake_label (label not used at test time)
    - TXT (negatives format): JSON with z_positions_m
- Saves predictions JSON/CSV; optionally copies positives (pred=1) to a subfolder.

Run:
  python lstm_infer_filelevel.py \
      --weights lstm_file_best_model.pt \
      --inputs Dataset/lstm_filelevel_samples \
      --out predictions_lstm_file \
      --copy_positives
"""

import os, json, glob, argparse, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# same TARGET_LEN and model as training
TARGET_LEN = 3000

def resample_to_len(z, target_len=TARGET_LEN):
    if len(z) == target_len:
        return z.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, num=len(z), endpoint=True)
    x_tgt = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    z_tgt = np.interp(x_tgt, x_src, z)
    return z_tgt.astype(np.float32)

def load_csv(fp):
    df = pd.read_csv(fp).sort_values("t_s").reset_index(drop=True)
    return df["z_m"].to_numpy(dtype=np.float32)

def load_txt(fp):
    with open(fp, "r") as f:
        d = json.load(f)
    return np.asarray(d["z_positions_m"], dtype=np.float32)

def standardize(z, scaler_mean, scaler_scale, mode):
    """
    If scaler stats are present (rich checkpoint), use their global average.
    Otherwise, fall back to per-file z-score.
    """
    if mode == "global" and scaler_mean is not None and scaler_scale is not None:
        mu = float(np.mean(scaler_mean))
        sd = float(np.mean(scaler_scale)) + 1e-9
    else:
        mu = float(z.mean())
        sd = float(z.std()) + 1e-9
    return (z - mu) / sd

class LSTMWakeFile(nn.Module):
    def __init__(self, input_size=1, hidden=128, num_layers=2, bidir=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidir,
                            dropout=dropout if num_layers>1 else 0.0)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Linear(out_dim, 1)
    def forward(self, x):  # x: (B,1,L)
        x = x.transpose(1,2)        # (B,L,1)
        h, _ = self.lstm(x)         # (B,L,H*)
        hmax = h.max(dim=1).values
        return self.head(hmax).squeeze(1)

def list_inputs(root):
    files = []
    for ext in ("*.csv", "*.txt"):
        files.extend(glob.glob(os.path.join(root, ext)))
        for sub in next(os.walk(root))[1]:
            files.extend(glob.glob(os.path.join(root, sub, ext)))
    files.sort()
    return files

def plot_predictions_grid(preds, out_dir, n=9, title="Sample Predictions (file-level)"):
    n = min(n, len(preds))
    idxs = random.sample(range(len(preds)), n)

    cols, rows = 3, (n + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8), squeeze=False)
    axes = axes.ravel()

    for ax_i, j in enumerate(idxs):
        fp = os.path.join(args.inputs, preds[j]["file"])
        prob = preds[j]["prob"]
        pred = preds[j]["pred"]

        # load data for plotting
        try:
            if fp.endswith(".csv"):
                df = pd.read_csv(fp).sort_values("t_s").reset_index(drop=True)
                t = df["t_s"].to_numpy(dtype=float)
                z = df["z_m"].to_numpy(dtype=float)
                w = df["wake_label"].to_numpy(dtype=int) if "wake_label" in df.columns else np.zeros_like(z, dtype=int)
            else:
                with open(fp, "r") as f:
                    d = json.load(f)
                z = np.asarray(d["z_positions_m"], dtype=float)
                dt = float(d.get("stepsize_ms", 200)) / 1000.0
                t = np.arange(len(z), dtype=float) * dt
                w = np.zeros_like(z, dtype=int)

            ax = axes[ax_i]
            ax.plot(t, z, lw=1.0)
            # shade wake regions
            if w.max() > 0:
                wpad = np.pad(w.astype(int), (1,1))
                dw = np.diff(wpad)
                starts = np.where(dw == 1)[0]
                ends   = np.where(dw == -1)[0]
                for a, b in zip(starts, ends):
                    ax.axvspan(t[a], t[b-1], color="red", alpha=0.25)

            ax.set_title(f"pred={pred} | p={prob:.2f}\n{os.path.basename(fp)}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        except Exception as e:
            axes[ax_i].set_title(f"Error: {os.path.basename(fp)}", fontsize=8)

    # Hide unused subplots
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    grid_path = os.path.join(out_dir, "predictions_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] Saved grid plot -> {grid_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # load checkpoint (model + scaler + threshold)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)  # <-- add weights_only=False
    model = LSTMWakeFile().to(device)

    # Try rich checkpoint first (dict with model_state + scaler + threshold)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        thr = float(ckpt.get("threshold", args.thr))
        scaler_mean = ckpt.get("scaler_mean", None)   # np.ndarray or None
        scaler_scale = ckpt.get("scaler_scale", None) # np.ndarray or None
        scale_mode = "global" if (scaler_mean is not None and scaler_scale is not None) else "perfile"
        print(f"[INFO] Loaded rich checkpoint. threshold={thr:.2f} scale_mode={scale_mode}")
    else:
        # Plain state_dict (e.g., saved via torch.save(model.state_dict(), ...))
        model.load_state_dict(ckpt)
        thr = float(args.thr)
        scaler_mean = None
        scaler_scale = None
        scale_mode = "perfile"
        print(f"[INFO] Loaded plain state_dict. Using per-file z-score & thr={thr:.2f}")

    model.eval()


    print(f"[INFO] Loaded: {args.weights} | best_threshold={thr:.2f}")

    files = list_inputs(args.inputs)
    if not files:
        print(f"[WARN] No files found in {args.inputs}")
        return
    print(f"[INFO] Found {len(files)} files")

    preds = []
    pos_dir = os.path.join(args.out, "positives")
    if args.copy_positives:
        os.makedirs(pos_dir, exist_ok=True)

    with torch.no_grad():
        for i, fp in enumerate(files, 1):
            ext = os.path.splitext(fp)[1].lower()
            if ext == ".csv":
                z = load_csv(fp)
            elif ext == ".txt":
                z = load_txt(fp)
            else:
                print(f"[SKIP] {fp} (unsupported)")
                continue

            z = resample_to_len(z, TARGET_LEN)
            z_sc = standardize(z, scaler_mean, scaler_scale, scale_mode)   # <-- use helper
            xb = torch.from_numpy(z_sc[None, None, :]).float().to(device)

            logit = model(xb)
            prob = torch.sigmoid(logit).item()
            pred = int(prob >= thr)

            preds.append({"file": os.path.relpath(fp, args.inputs).replace("\\","/"),
                          "prob": float(prob), "pred": pred})

            if args.copy_positives and pred == 1:
                dest = os.path.join(pos_dir, os.path.basename(fp))
                shutil.copy2(fp, dest)

            if i % 50 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {fp} -> pred={pred} p={prob:.2f}")

    # save outputs
    out_json = os.path.join(args.out, "predictions.json")
    with open(out_json, "w") as f:
        json.dump(preds, f, indent=2)
    out_csv = os.path.join(args.out, "predictions.csv")
    pd.DataFrame(preds).to_csv(out_csv, index=False)
    print(f"[DONE] JSON -> {out_json}")
    print(f"[DONE] CSV  -> {out_csv}")
    if args.copy_positives:
        print(f"[DONE] copied positives -> {pos_dir}")
    plot_predictions_grid(preds, args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to lstm_file_best.pt")
    ap.add_argument("--inputs",  type=str, required=True, help="Folder with new CSV/TXT files")
    ap.add_argument("--out",     type=str, default="predictions_lstm_file")
    ap.add_argument("--copy_positives", action="store_true", help="Copy predicted positives to out/positives/")
    ap.add_argument("--thr", type=float, default=0.5, help="Decision threshold if checkpoint has none (plain state_dict)")

    args = ap.parse_args()
    main(args)
