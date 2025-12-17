# Timeseries Dataset Preparation Pipeline

This repository contains utilities for converting raw vessel wake detection data
(images, labels, and timeseries `.txt` files) into a **processed format suitable for
anomaly detection or timeseries modeling**.

---

## 📂 Project Structure

```

__marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac
   ├── data
   │   ├── metaData.csv                # Metadata (Excel/CSV)
   │   └── vessel_wake_timeseries_data
   │       ├── train                   # Raw timeseries text files
   │       ├── test
   │       └── valid
   └── scripts                         # Utility scripts

To_share_with_marine_lab/
   ├──Dataset
       ├── images/{train,test,valind}         # Spectrogram/frequency-domain images (.jpg)
       ├── labels/{train,test,valind}         # Corresponding YOLO-format labels (.txt)
   ├──scripts
       ├── prepare_timeseries_dataset.py   # Main preprocessing script
       └── README.md                       # (this file)
```

---

## 🚀 Usage

### Convert all text files into CSV (no windowing)
This preserves raw signals, mapped with labels, for baseline anomaly detection.

```bash
python prepare_timeseries_dataset.py   --labels_root ../Dataset/labels   --timeseries_root ../../marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac/data/vessel_wake_timeseries_data   --out_root processed_ts
```

---

## 📑 Output

- For each split (`train`, `valid`, `test`), an `_index.csv` is created with:
  - `key`: the unique identifier (basename + timestamp)
  - `out_csv`: path to the processed CSV file
  - `label`: positive/negative label
- Individual CSV files contain the raw timeseries for each labeled example.

Example:

```
processed_ts/
  ├── train/
  │   ├── _index.csv
  │   ├── bcip_04-1634226600.csv
  │   └── ...
  ├── valid/
  ├── test/
```

---

## ⚙️ Implementation Notes

- **No filename assumptions**: The script matches files by shared prefix up to the
  timestamp, not by strict template.
- **Label alignment**: Only timeseries with matching labels are processed.
- **Multiple timeseries**: If multiple exist for a given key (e.g. `-timeseries-1.txt`,
  `-timeseries-2.txt`), the one with label is used.
- **Validation split**: All splits (`train`, `valid`, `test`) are processed consistently.

---

## 🔍 Expected Logs

- `[info] ... keys have labels but no timeseries` → means some image/label pairs do not have raw timeseries available (skipped).
- `[warn] multiple timeseries for key ...; using first` → normal if dataset includes multiple recordings for the same timestamp.

---

## 👩‍💻 Maintainers

- Original author: Sheida Majouni  
- Purpose: Vessel wake detection (DeepSense / MarineLabs dataset)  
- Updated: 2025


