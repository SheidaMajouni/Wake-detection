#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import pandas as pd
import boto3

TRAINING_META_BUCKET = "marinelabs-data-science"

def fetch_training_data(
    wake_meta_path: str,
    background_meta_path: str,
    output_path: Path,
    overwrite_existing: bool
):
    s3 = boto3.client('s3')
    
    wake_df = pd.read_csv(wake_meta_path)
    background_df = pd.read_csv(background_meta_path)
    df = pd.concat((wake_df, background_df))

    # initialize output directories
    cv_categories = ['train', 'test', 'valid']
    image_dst, label_dst = {}, {}
    for category in cv_categories:
        image_dst[category] = output_path / f"images/{category}"
        image_dst[category].mkdir(parents=True, exist_ok=True)
        label_dst[category] = output_path / f"labels/{category}"
        label_dst[category].mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        if row.category == 'none':
            continue
        assert(row.category in cv_categories)

        response = s3.list_objects_v2(
            Bucket=TRAINING_META_BUCKET, 
            Prefix=f"vessel_wake_training_data/{row.deployment_id}/{row.report_cycle_start_epoch}"
        )
        for obj in response['Contents']:
            if obj['Key'].endswith(".txt"):
                label_key = obj['Key']
                print(label_key)

                # deployment_id = box_key.split("/")[1]
                # report_cycle_start_epoch = int(box_key.split("/")[2])
                # sub_df = df[(df.deployment_id == deployment_id) & (df.report_cycle_start_epoch == report_cycle_start_epoch)]
                # category = sub_df.category.iloc[0]    

                meta_key = label_key.replace(".txt",".jsn")
                meta_obj = s3.get_object(Bucket=TRAINING_META_BUCKET, Key=meta_key)
                meta_dict = json.loads(meta_obj['Body'].read())

                # download image and bounding box files
                label_fullpath = label_dst[row.category] / Path(label_key).name
                if overwrite_existing or (not label_fullpath.exists()):
                    print(f"downloading {str(label_fullpath)}")
                    s3.download_file(
                        TRAINING_META_BUCKET, 
                        label_key,
                        str(label_fullpath)
                    )

                image_fullpath = image_dst[row.category] / Path(meta_dict['image_key']).name
                if overwrite_existing or (not image_fullpath.exists()):
                    print(f"downloading {str(image_fullpath)}")
                    s3.download_file(
                        meta_dict['image_bucket'], 
                        meta_dict['image_key'], 
                        str(image_fullpath)
                    )
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
        type=Path,
        action='store_true',
        default=False,
        help="If specified, existing labels and images will be overwritten"
    )

    args = parser.parse_args()
    params = {}
    for key, value in vars(args).items():
        params[key] = value

    fetch_training_data(
        wake_meta_path=args.wake_meta_path,
        background_meta_path=args.background_meta_path,
        output_path=args.output_path,
        overwrite_existing=args.overwrite_existing
    )

    print("\nDone!")
