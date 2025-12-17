#!/usr/bin/env python3

# Download image files from an existing location in S3, providing environment
# (e.g. production/development), deployment_id, and cycle_start_epoch (epoch timestamp)
# as parameters.
# See detailed usage notes at bottom, or by typing `download_images.py --help`
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import boto3

S3_BUCKET = "marinelabs-data"
S3_PATH_TEMPLATE = "{environment}/{deployment_id}/{cycle_epoch}/images/"


def _get_cycle_epoch_list(environment: str, deployment_id: str, time_range: Tuple[int, int]):
    cycle_epochs = []
    start, end = time_range

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    s3_prefix = f"{environment}/{deployment_id}/"

    for digit1, digit2 in zip(str(start), str(end)):
        if digit1 == digit2:
            s3_prefix += digit1
        else:
            break

    for response in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix, Delimiter="/"):
        for item in response.get("CommonPrefixes", []):
            try:
                s3_path = item["Prefix"]
                epoch = int(s3_path.rsplit("/", 2)[-2])
                if start <= epoch <= end:
                    cycle_epochs.append(epoch)
            except ValueError:
                # Ignore "epochs" that aren't an integer. In practice this shouldn't ever happen,
                # but there could be a case where an S3 path was incorrectly set in development
                continue

    return cycle_epochs


def fetch_images(
    environment: str,
    deployment_id: str,
    output_path: Path,
    time_range: Optional[Tuple[int, int]] = None,
    timestamp: Optional[int] = None,
    json_file: Optional[Path] = None,
):
    if timestamp:
        cycle_epochs = [timestamp]

    elif json_file:
        cycle_epochs = json.loads(json_file.read_text())

    elif time_range:
        cycle_epochs = _get_cycle_epoch_list(environment, deployment_id, time_range)

    else:
        print("Exactly one of time_range, timestamp, or json_file must be supplied")
        exit(1)

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for cycle_epoch in cycle_epochs:
        s3_path = S3_PATH_TEMPLATE.format(
            environment=environment, deployment_id=deployment_id, cycle_epoch=cycle_epoch
        )
        local_filename_prefix = f"{deployment_id}-{cycle_epoch}-image"

        for response in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_path):
            for item in response.get("Contents", []):
                s3_key = item["Key"]
                local_filename = f"{local_filename_prefix}-{s3_key.rsplit('/', 1)[-1]}"
                local_file_path = output_path / local_filename

                if local_file_path.exists():
                    print(f"Avoiding downloading {local_file_path} as it already exists.")
                else:
                    s3.download_file(Bucket=S3_BUCKET, Key=s3_key, Filename=str(local_file_path))
                    print(f"Output file saved to {local_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images for offline inspection.")
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        choices=("production", "staging", "development"),
        default="production",
        help="The environment that you want to download images from (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--deployment-id",
        type=str,
        required=True,
        help="The ID of the buoy whose image data you are downloading (e.g. prince_rupert_03).",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path(__file__).absolute().parent / "images",
        help="Specify a path for the downloaded file(s).",
    )

    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--time-range",
        type=int,
        nargs=2,
        help="The start and end time range (in Epoch seconds, inclusive) of images to download.",
    )
    selection_group.add_argument(
        "--timestamp",
        type=int,
        help=(
            "The UNIX timestamp (seconds since epoch) of the cycle (aka cycle_start_epoch) you are "
            "downloading data for."
        ),
    )
    selection_group.add_argument(
        "-j",
        "--json-file",
        type=Path,
        help="JSON filename, containing a list of report epochs to fetch images for.",
    )

    args = parser.parse_args()
    params = {}
    for key, value in vars(args).items():
        params[key] = value

    fetch_images(
        environment=args.environment,
        deployment_id=args.deployment_id,
        output_path=args.output_path,
        time_range=args.time_range,
        timestamp=args.timestamp,
        json_file=args.json_file,
    )

    print("\nDone!")
