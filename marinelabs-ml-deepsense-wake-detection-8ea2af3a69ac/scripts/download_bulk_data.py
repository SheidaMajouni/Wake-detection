#!/usr/bin/env python3
import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import pandas as pd

from ml_deepsense_wake_detection.es_client import FIELDS, ElasticsearchClient

METRIC_SOURCE_MAP = {
    metric: source for key in FIELDS for metric, source in [key.rsplit(".", 1)]
}


def download_buoy_data(
    environment: str,
    username: str,
    password: str,
    deployment_id: str,
    output_directory: Path,
    time_range: Optional[Tuple[int, int]] = None,
    timestamp: Optional[int] = None,
    json_output: bool = False,
):
    es_client = ElasticsearchClient(
        environment=environment, username=username, password=password
    )

    if timestamp is not None:
        filename = f"{deployment_id}_{timestamp}"
        es_document = es_client.get_document(
            deployment_id=deployment_id, timestamp=timestamp
        )

        if es_document is None:
            es_documents = []
        else:
            es_documents = [
                es_client.get_document(deployment_id=deployment_id, timestamp=timestamp)
            ]

    elif time_range is not None:
        start, end = time_range
        filename = f"{deployment_id}_from_{start}_to_{end}"
        es_documents = list(
            es_client.document_search(deployment_id=deployment_id, start=start, end=end)
        )

    else:
        raise ValueError("Exactly one of [time_range, timestamp] must be supplied.")

    num_documents = len(es_documents)
    print(f"Found {num_documents} {'document' if num_documents == 1 else 'documents'}.")

    if num_documents == 0:
        # No Elasticsearch documents were found, so we can exit early
        return

    # Ensure that the output directory exists
    output_directory.mkdir(exist_ok=True, parents=True)

    def parse_es_documents(es_documents) -> Iterator[dict[str, Any]]:
        for es_document in es_documents:
            yield {
                metric: es_document[metric][source]
                for metric, source in METRIC_SOURCE_MAP.items()
            }

    parsed_es_documents = parse_es_documents(es_documents)

    if json_output:
        filename = f"{filename}.json"
        output_file = output_directory / filename
        print(f"Writing output to: {output_file}")

        with output_file.open("w") as output:
            json.dump(list(parsed_es_documents), output, indent=2)

    else:
        filename = f"{filename}.csv"
        output_file = output_directory / filename
        print(f"Writing output to: {output_file}")

        with output_file.open("w") as output:
            writer = csv.DictWriter(output, list(METRIC_SOURCE_MAP))
            writer.writeheader()
            writer.writerows(
                {key: str(value) for key, value in es_document.items()}
                for es_document in parsed_es_documents
            )

        df = pd.read_csv(output_file)
        df = df.sort_values(["report_cycle_start_epoch"])
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Download time series buoy data for offline inspection. By default result data is "
            "returned in CSV format."
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
        "-d",
        "--deployment-id",
        type=str,
        required=True,
        help=(
            "The ID of the buoy whose time series data you are downloading (e.g. prince_rupert_03)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).absolute().parent / "time_series_data",
        help="Specify a directory for the downloaded time series. Default is %(default)s",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output data in JSON format rather than CSV format.",
    )

    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--time-range",
        type=int,
        nargs=2,
        help=(
            "The start and end time range (in Epoch seconds, inclusive) of time series data "
            "to download."
        ),
    )
    selection_group.add_argument(
        "--timestamp",
        type=int,
        help=(
            "The UNIX timestamp (seconds since Epoch) of the cycle (aka cycle_start_epoch) you are "
            "downloading data for."
        ),
    )

    args = parser.parse_args()
    download_buoy_data(
        environment=args.environment,
        username=args.username,
        password=args.password,
        deployment_id=args.deployment_id,
        output_directory=args.output,
        time_range=args.time_range,
        timestamp=args.timestamp,
        json_output=args.json,
    )

    print("\nDone!")
