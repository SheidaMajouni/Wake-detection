import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd

# Key: <prefix>-<epoch>   (first 10–13 digit epoch)
EPOCH_KEY_RE = re.compile(r"^(?P<prefix>.+?)-(?P<epoch>\d{10,13})\b", re.ASCII)

# Label variant patterns
LBL_VARIANTS = [
    re.compile(r"-spectral-(?P<vid>\d+)\b"),
    re.compile(r"-wake-cwt_image_(?P<vid>\d+)\b"),
    re.compile(r"-timeseries-(?P<vid>\d+)\b"),
]
# or 
# VARIANT_RE = re.compile(r"-(?P<type>[a-zA-Z_]+)-(?P<vid>\d+)\b")

# Timeseries variant pattern
TS_VARIANT = re.compile(r"-timeseries-(?P<vid>\d+)\b")

def extract_key_and_variant(p: Path, is_label: bool) -> tuple[str|None, int|None]:
    """
    Return (key, variant_index) from filename stem.
    key = '<prefix>-<epoch>' based on first 10–13 digit epoch.
    variant_index = trailing index (e.g., spectral-2, wake-cwt_image_2, timeseries-2).
    """
    stem = p.stem
    m = EPOCH_KEY_RE.match(stem)
    if not m:
        return None, None
    key = f"{m.group('prefix')}-{m.group('epoch')}"

    vid = None
    if is_label:
        for rx in LBL_VARIANTS:
            mm = rx.search(stem)
            if mm:
                vid = int(mm.group("vid"))
                break
    else:
        mm = TS_VARIANT.search(stem)
        if mm:
            vid = int(mm.group("vid"))

    return key, vid

def load_timeseries_json(path: str | Path):
    with open(path, "r") as f:
        obj = json.load(f)
    start = int(obj["start_epoch_ms"])
    n = int(obj["num_timesteps"])
    step = int(obj["stepsize_ms"])
    z = obj["z_positions_m"]
    if len(z) != n:
        raise ValueError(f"len(z)={len(z)} != num_timesteps={n} in {path}")
    t_ms = np.arange(n, dtype=np.int64) * step + start
    df = pd.DataFrame({
        "timestamp_ms": t_ms,
        "t_s": (t_ms - start)/1000.0,
        "z_m": z
    })
    meta = {
        "start_epoch_ms": start,
        "num_timesteps": n,
        "stepsize_ms": step,
        "duration_ms": (n-1)*step
    }
    return df, meta

def load_yolo_labels(path: str | Path):
    """Return list of dicts: {cls:int, xc:float, yc:float, w:float, h:float}."""
    labels = []
    if not Path(path).exists():
        return labels
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            labels.append({
                "cls": int(float(cls)),
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w),
                "h": float(h),
            })
    return labels

def yolo_time_window(xc: float, w: float, ts_meta: dict, clamp=True):
    dur = ts_meta["duration_ms"]
    start = ts_meta["start_epoch_ms"]
    t0 = start + (xc - w/2.0) * dur
    t1 = start + (xc + w/2.0) * dur
    if clamp:
        t0 = max(start, t0)
        t1 = min(start + dur, t1)
    return float(t0), float(t1)

def label_series_by_windows(df: pd.DataFrame, windows_ms: list[tuple[float,float]]):
    lab = np.zeros(len(df), dtype=np.int8)
    t = df["timestamp_ms"].to_numpy()
    for (t0, t1) in windows_ms:
        mask = (t >= t0) & (t <= t1)
        lab[mask] = 1
    out = df.copy()
    out["wake_label"] = lab
    return out

def process_split(labels_split_dir: Path,
                  timeseries_split_dir: Path,
                  out_split_dir: Path,
                  event_class_ids=(0,),
                  save_individual_csv=True):
    """
    For each LABEL file:
      * compute (key, label_variant)
      * pick timeseries with same key AND same variant if available
        - else prefer variant==1
        - else fallback to the smallest available variant for that key
      * convert + label and write <key>-<label_variant>-labeled.csv
    Only processes items that have labels.
    """
    out_split_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_split_dir.glob("*.txt"))
    if not label_files:
        print(f"[info] No label .txt files in {labels_split_dir}")
        idx_path = out_split_dir / "_index.csv"
        pd.DataFrame(columns=[
            "key","label_variant","label_file","timeseries_file","out_csv",
            "n_points","n_positive","positive_frac","start_epoch_ms","stepsize_ms","duration_ms","n_boxes"
        ]).to_csv(idx_path, index=False)
        return pd.DataFrame(), str(idx_path)

    # Build timeseries index: key -> {variant:int -> Path}
    ts_index: dict[str, dict[int, Path]] = {}
    for p in sorted(timeseries_split_dir.glob("*.txt")):
        key, vid = extract_key_and_variant(p, is_label=False)
        if not key:
            continue
        if vid is None:
            # if no explicit timeseries variant, treat as 1
            vid = 1
        ts_index.setdefault(key, {})[vid] = p

    index_rows = []
    for lbl in label_files:
        key, lbl_vid = extract_key_and_variant(lbl, is_label=True)
        if not key:
            # no epoch in label filename -> skip
            print(f"[warn] cannot extract key from label: {lbl.name}")
            continue
        if lbl_vid is None:
            # if label has no explicit variant, assume 1
            lbl_vid = 1

        ts_candidates = ts_index.get(key, {})
        if not ts_candidates:
            # no timeseries for this key -> skip
            # (if you want to fabricate all-zero labels, we can add a flag)
            continue

        # Choose best timeseries variant:
        # 1) exact match
        # 2) variant 1
        # 3) smallest available
        if lbl_vid in ts_candidates:
            ts_path = ts_candidates[lbl_vid]
        elif 1 in ts_candidates:
            ts_path = ts_candidates[1]
        else:
            min_vid = min(ts_candidates.keys())
            ts_path = ts_candidates[min_vid]
            if lbl_vid != min_vid:
                print(f"[info] using nearest ts variant {min_vid} for label variant {lbl_vid} on key {key}")

        # Load YOLO boxes (keep target classes)
        boxes = [b for b in load_yolo_labels(lbl) if b["cls"] in event_class_ids]

        # Convert
        df_ts, meta = load_timeseries_json(ts_path)
        windows = [yolo_time_window(b["xc"], b["w"], meta) for b in boxes]
        df_lab = label_series_by_windows(df_ts, windows)

        out_csv = out_split_dir / f"{key}-v{lbl_vid}-labeled.csv"
        if save_individual_csv:
            df_lab.to_csv(out_csv, index=False)

        index_rows.append({
            "key": key,
            "label_variant": int(lbl_vid),
            "label_file": str(lbl),
            "timeseries_file": str(ts_path),
            "out_csv": str(out_csv),
            "n_points": len(df_lab),
            "n_positive": int(df_lab["wake_label"].sum()),
            "positive_frac": float(df_lab["wake_label"].mean()),
            "start_epoch_ms": int(meta["start_epoch_ms"]),
            "stepsize_ms": int(meta["stepsize_ms"]),
            "duration_ms": int(meta["duration_ms"]),
            "n_boxes": len(windows),
        })

    cols = ["key","label_variant","label_file","timeseries_file","out_csv",
            "n_points","n_positive","positive_frac","start_epoch_ms","stepsize_ms","duration_ms","n_boxes"]
    idx = pd.DataFrame(index_rows, columns=cols)
    if not idx.empty:
        idx = idx.sort_values(["key","label_variant"])
    idx_path = out_split_dir / "_index.csv"
    idx.to_csv(idx_path, index=False)
    return idx, str(idx_path)

def run_pipeline(labels_root, timeseries_root, out_root, splits=("train","val","test"), event_class_ids=(0,)):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    for split in splits:
        labels_dir = Path(labels_root) / split
        ts_dir     = Path(timeseries_root) / split
        out_dir    = out_root / split
        idx, idx_csv = process_split(labels_dir, ts_dir, out_dir, event_class_ids=event_class_ids)
        split_summary[split] = {
            "index_csv": str(idx_csv),
            "n_series": int(len(idx)),
            "avg_pos_frac": float(idx["positive_frac"].mean()) if len(idx) else 0.0
        }
    return split_summary

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert raw wake time series to labeled CSVs by matching <prefix>-<epoch> and aligning variant indices (no windowing).")
    p.add_argument("--labels_root", required=True, help="Path to __Dataset/labels")
    p.add_argument("--timeseries_root", required=True, help="Path to marinelabs/.../vessel_wake_timeseries_data")
    p.add_argument("--out_root", required=True, help="Where to write labeled CSVs")
    p.add_argument("--splits", default="train,valid,test", help="Comma-separated splits (default: train,valid,test)")
    p.add_argument("--event_class_ids", default="0", help="Comma-separated YOLO class ids as positive (default: '0')")
    args = p.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    event_cls = tuple(int(x) for x in args.event_class_ids.split(",") if x.strip())

    summary = run_pipeline(
        labels_root=args.labels_root,
        timeseries_root=args.timeseries_root,
        out_root=args.out_root,
        splits=splits,
        event_class_ids=event_cls
    )
    print(pd.DataFrame(summary).T)
    
# how to run
# python prepare_timeseries_dataset.py   --images_root Dataset/images   --labels_root Dataset/labels   --timeseries_root marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac/data/vessel_wake_timeseries_data   --out_root processed_ts 
