#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd
import json
from pathlib import Path
import csv
import boto3

from ml_deepsense_wake_detection.es_client import ElasticsearchClient

BUCKET = "marinelabs-data-science"
S3_PATH = "vessel_wake_training_data"
KEYS = [
    'deployment_id',
    'report_cycle_start_epoch',
    'box_x_centers',
    'box_y_centers',
    'box_widths',
    'box_heights',
    'service_state',
    'wave_quality',
    'wave_significant_height_m',
    'wave_max_height_m',
    'wave_peak_period_s',
    'cwt_start_time_ms',
    'cwt_end_time_ms',
    'cwt_min_frequency',
    'cwt_max_frequency',
]

def fetch_training_meta(
    environment: str,
    username: str,
    password: str,
    output_path: Path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # initalize ES client
    es_client = ElasticsearchClient(environment=environment, username=username, password=password)

    # create empty output file
    with open(str(output_path), 'w') as f:
        csvwriter = csv.DictWriter(f, KEYS)
        csvwriter.writeheader()

    # loop thru all keys in vessel wake training path
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=S3_PATH)
    for page in pages:
        for obj in page['Contents']:
            if "manually_processed_images" in obj['Key']:
                continue
            if not obj['Key'].endswith(".txt"):
                continue

            print(obj['Key'])

            meta_dict = {
                'deployment_id': obj['Key'].split("/")[-3],
                'report_cycle_start_epoch': obj['Key'].split("/")[-2]
            }

            # get bounding box info
            box_obj = s3.get_object(
                Bucket=BUCKET,
                Key=obj['Key']
            )
            box_df = pd.read_csv(box_obj['Body'], sep='\s+', names=('class','x','y','w','h'))
            meta_dict['box_x_centers'] = box_df.x.tolist()
            meta_dict['box_y_centers'] = box_df.y.tolist()
            meta_dict['box_widths'] = box_df.w.tolist()
            meta_dict['box_heights'] = box_df.h.tolist()

            # get CWT image location from box meta
            box_meta_obj = s3.get_object(
                Bucket=BUCKET,
                Key=obj['Key'].replace(".txt",".jsn")
            )
            box_meta = json.loads(box_meta_obj['Body'].read())

            # get wave height/period data from ES
            es_doc = es_client.get_document(
                deployment_id=meta_dict['deployment_id'],
                timestamp=meta_dict['report_cycle_start_epoch']
            )

            for key in ['service_state','wave_quality','wave_significant_height_m',
                        'wave_max_height_m', 'wave_peak_period_s']:
                try:
                    meta_dict[key] = es_doc[key]
                except KeyError:
                    meta_dict[key] = None

            # load metadata for CWT image
            if "manually_processed_images" in box_meta['image_key']:
                # metadata will be stored in JSON file with same name
                image_meta_obj = s3.get_object(
                    Bucket=box_meta['image_bucket'],
                    Key=box_meta['image_key'].replace(".jpg",".jsn")
                )
                image_meta = json.loads(image_meta_obj['Body'].read())
                meta_dict['cwt_start_time_ms'] = image_meta['start_time']*1000
                meta_dict['cwt_end_time_ms'] = image_meta['end_time']*1000
                meta_dict['cwt_min_frequency'] = image_meta['min_frequency']
                meta_dict['cwt_max_frequency'] = image_meta['max_frequency']
            else:
                # metadata will be stored in marinelabs-data bucket in file ending with "cwt_*.txt"
                image_meta_key = box_meta['image_key'].replace(".jpg",".txt").replace("cwt_image","cwt")
                image_meta_obj = s3.get_object(
                    Bucket=box_meta['image_bucket'],
                    Key=box_meta['image_key'].replace(".jpg",".txt").replace("cwt_image","cwt")
                )
                image_meta = json.loads(image_meta_obj['Body'].read())['meta']
                meta_dict['cwt_start_time_ms'] = image_meta['start_epoch_ms']
                meta_dict['cwt_end_time_ms'] = image_meta['end_epoch_ms']
                meta_dict['cwt_min_frequency'] = image_meta['min_frequency_hz']
                meta_dict['cwt_max_frequency'] = image_meta['max_frequency_hz']

            with open(str(output_path), 'a') as f:
                csvwriter = csv.DictWriter(f, KEYS)
                csvwriter.writerow(meta_dict)

    return

if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Download metadata for all available vessel wake training data in CSV format."
        )
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        choices=("production", "staging", "development"),
        default="production",
        help="The environment that you want to download images from (default: %(default)s)",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        required=True,
        help="The username used to authenticate with Elasticsearch.",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        required=True,
        help="The password used to authenticate with Elasticsearch.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Specify a filename for the downloaded metadata.",
    )

    args = parser.parse_args()
    fetch_training_meta(
        environment=args.environment,
        username=args.username,
        password=args.password,
        output_path=args.output
    )

    print("\nDone!")
