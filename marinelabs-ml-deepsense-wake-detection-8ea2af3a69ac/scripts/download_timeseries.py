#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import pandas as pd
import boto3

S3_BUCKET = "marinelabs-data"
S3_PATH_TEMPLATE = "{environment}/{deployment_id}/{cycle_epoch}/spectral/"

def fetch_timeseries(
    wake_meta_path: str,
    background_meta_path: str,
    output_path: Path,
    overwrite_existing: bool,
    environment: str='production'
):
    wake_df = pd.read_csv(wake_meta_path)
    background_df = pd.read_csv(background_meta_path)
    df = pd.concat((wake_df, background_df))

    # initialize output directories
    cv_categories = ['train', 'test', 'valid']
    dst = {}
    for category in cv_categories:
        dst[category] = output_path / f"{category}"
        dst[category].mkdir(parents=True, exist_ok=True)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator("list_objects_v2")

    for idx, row in df.iterrows():
        if row.category == 'none':
            continue
        assert(row.category in cv_categories)

        s3_path = S3_PATH_TEMPLATE.format(
            environment=environment,
            deployment_id=row.deployment_id,
            cycle_epoch=row.report_cycle_start_epoch
        )
        local_filename_prefix = f"{row.deployment_id}-{row.report_cycle_start_epoch}-timeseries"

        for response in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_path):
            for item in response.get("Contents", []):
                s3_key = item["Key"]
                local_filename = f"{local_filename_prefix}-{s3_key.rsplit('/', 1)[-1]}"
                local_file_path = dst[row.category] / local_filename

                if (not overwrite_existing) and local_file_path.exists():
                    print(f"Avoiding downloading {local_file_path} as it already exists.")
                else:
                    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                    data = json.loads(obj['Body'].read())
                    with open(local_file_path, 'w') as f:
                        json.dump(data['timeseries'], f)
                    print(f"Output file saved to {local_file_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download vessel wake training images and bounding box labels.")
    parser.add_argument(
        "-w",
        "--wake_meta_path",
        type=Path,
        required=True,
        help="Path to CSV file containing metadata for vessel wake training images",
    )
    parser.add_argument(
        "-b",
        "--background_meta_path",
        type=Path,
        required=True,
        help="Path to CSV file containing metadata for background training images",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Output path for labels and training images"
    )
    parser.add_argument(
        "--overwrite_existing",
        action='store_true',
        default=False,
        help="If specified, existing labels and images will be overwritten"
    )

    args = parser.parse_args()
    params = {}
    for key, value in vars(args).items():
        params[key] = value

    fetch_timeseries(
        wake_meta_path=args.wake_meta_path,
        background_meta_path=args.background_meta_path,
        output_path=args.output_path,
        overwrite_existing=args.overwrite_existing
    )

    print("\nDone!")
