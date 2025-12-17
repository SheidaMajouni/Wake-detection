#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import ast
import numpy as np
import random
import matplotlib.pyplot as plt

BIN_KEYS = [
    'num_wakes',
    'max_widths',
    'max_heights',
    'wave_max_height_m',
    'wave_significant_height_m',
    'wave_peak_period_s'
]
BACKGROUND_BIN_KEYS = [
    'wave_max_height_m',
    'wave_significant_height_m',
    'wave_peak_period_s'
]


def assign_by_bin(df, sub_df, key_idx, key_list, test_ratio, valid_ratio, num_bins=3):
    n_rows = len(sub_df)
    if (n_rows < 3*num_bins) or (key_idx >= len(key_list)):
        # print(f"{n_rows} {key_idx} assigning category")
        n_test = np.maximum(round(n_rows * test_ratio), 1)
        n_valid = np.maximum(round(n_rows * valid_ratio), 1)
        n_train = n_rows - n_test - n_valid

        # check if any rows in sub_df need to be excluded from train/validate sets
        indices = sub_df.index.tolist()
        random.shuffle(indices)
        df.loc[indices[0:n_train], 'category'] = "train"
        df.loc[indices[n_train:n_train+n_test], 'category'] = "test"
        df.loc[indices[n_train+n_test:n_train+n_test+n_valid], 'category'] = "valid"
    else:
        key = key_list[key_idx]
        # print(f"{n_rows} {key_idx} {key} performing qcut")
        bins = pd.qcut(sub_df[key], num_bins, duplicates='drop')
        for b in bins.unique():
            # print(f"assign_by_bin {b}")
            df = assign_by_bin(df, sub_df[bins == b], key_idx+1, key_list, test_ratio, valid_ratio)
    return df

def split_training_data(
    meta_path: Path,
    output_dir: Path,
    test_ratio: float,
    valid_ratio: float
):
    assert(test_ratio + valid_ratio < 1.0)

    # load training metadata
    full_df = pd.read_csv(str(meta_path))
    full_df.loc[full_df.service_state.isna(), 'service_state'] = 1 # assume all data with no service state is valid
    full_df.loc[full_df.wave_quality.isna(), 'wave_quality'] = 1 # assume all data with no wave quality is valid
    full_df = full_df[(full_df.service_state == 1) & (full_df.wave_quality == 1)]
    full_df = full_df.dropna()
    for k in ['box_x_centers', 'box_y_centers', 'box_widths', 'box_heights']:
        full_df[k] = full_df[k].apply(lambda x: ast.literal_eval(x))
    full_df.loc[:,'num_boxes'] = [len(x) for x in full_df.box_x_centers]
    full_df['num_wakes'] = full_df['num_boxes']

    # split into wake and background datasets
    wake_df = full_df[full_df.num_boxes > 0]
    background_df = full_df[full_df.num_boxes == 0]
    wake_df['category'] = "none"
    wake_df = wake_df.reset_index(drop=True)
    background_df['category'] = "none"
    background_df = background_df.reset_index(drop=True)

    print(f"images with wakes: {len(wake_df)} of {len(full_df)}")

    # add additional meta to wake dataframe
    min_widths, max_widths = [], []
    min_heights, max_heights = [], []
    for idx, row in wake_df.iterrows():
        min_widths.append(np.min(row['box_widths']))
        max_widths.append(np.max(row['box_widths']))
        min_heights.append(np.min(row['box_heights']))
        max_heights.append(np.max(row['box_heights']))
    wake_df.loc[:,'min_widths'] = min_widths
    wake_df.loc[:,'max_widths'] = max_widths
    wake_df.loc[:,'min_heights'] = min_heights
    wake_df.loc[:,'max_heights'] = max_heights

    # assign CV categories
    wake_df = assign_by_bin(wake_df, wake_df, 0, BIN_KEYS, test_ratio, valid_ratio)
    assert(not any(wake_df.category == 'none'))
    background_df = assign_by_bin(background_df, background_df, 0, BACKGROUND_BIN_KEYS, test_ratio, valid_ratio)
    assert(not any(background_df.category == 'none'))

    output_dir.mkdir(parents=True, exist_ok=True)
    wake_df.to_csv(str(output_dir / "wake_meta.csv"))
    background_df.to_csv(str(output_dir / "background_meta.csv"))

    # plot resulting distributions
    for plot_key in BACKGROUND_BIN_KEYS:
        ax = background_df.pivot(columns='category')[plot_key][['train','valid','test']].plot(kind = 'hist', stacked=True, figsize=(12,8))
        ax.set_title("Background " + plot_key)
        # ax.set_yscale('log')
    for plot_key in BIN_KEYS:
        ax = wake_df.pivot(columns='category')[plot_key][['train','valid','test']].plot(kind = 'hist', stacked=True, figsize=(12,8))
        ax.set_title(plot_key)
        # ax.set_yscale('log')
    for plot_key in ['num_wakes', 'deployment_id']:
        ax = wake_df.groupby([plot_key, 'category']).size().unstack()[['train','valid','test']].plot(kind='bar', stacked=True, figsize=(12,8))
        ax.set_title(plot_key)
        # ax.set_yscale('log')
    plt.figure()
    ax = wake_df.category.value_counts().plot(kind='bar', figsize=(12,8))
    plt.show()

    return

if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Download metadata for all available vessel wake training data in CSV format."
        )
    )
    parser.add_argument(
        "-m",
        "--meta_path",
        type=Path,
        required=True,
        help="Path to CSV file containing metadata for training images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for wake metadata (wake_meta.csv) and background metadata (background_meta.csv)"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        required=True,
        help="Ratio of data to use for validation"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        required=True,
        help="Ratio of data to use for testing"
    )

    args = parser.parse_args()
    split_training_data(
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio
    )

    print("\nDone!")
