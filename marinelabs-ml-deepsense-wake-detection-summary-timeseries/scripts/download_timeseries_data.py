#!/usr/bin/env python3
import csv
import pandas as pd
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from elasticsearch import Elasticsearch, NotFoundError


METRICS = (
    "report_cycle_start_epoch.primary",
    "wave_max_height_m.primary",
    "wave_peak_direction_degrees.primary",
    "wave_peak_period_s.primary",
    "wave_significant_height_m.primary",
    "wave_mean_direction_degrees.primary",
    "wave_mean_period_m2_s.primary",
    "wave_mean_period_m1_s.primary",
    "wave_mean_period_mn1_s.primary",
    "wave_quality.primary",
    "wind_gust_kn.cloud",
    "wind_direction_degrees.cloud",
    "wind_speed_kn.cloud",
    "wind_quality.primary",
    "air_humidity_percent.bme",
    "air_pressure_pa.bme",
    "air_temperature_c.bme",
    "air_humidity_heat_compensated_percent.primary",
    "air_temperature_heat_compensated_c.primary",
)

_CLOUD_ID_MAP = {
    "development": "development:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkZTRjOTEzMjI0YzEyNGI3NThmNzFkY2ZlNjVhNjcxYmUkMjI2MWI1MzJlN2I0NDMwMGIzZTE1MGM0Y2RiM2RhNmQ=",  # NOQA: E501
    "staging": "staging:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkZTY3NWQ0ZDRjZTE2NDFjZjgwZDU0NjIxMWM1NGY3ZjAkMWM3MjQ2OWNhMjUwNGNjZGFlMjc3MTNiNTQyNDMyZjk=",  # NOQA: E501
    "production": "production:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkMjQ4ZGUxMGFiZWQzNDZhNjlhOTZkMDBhMTJiZTJmOGEkNWQwNjRhYjczZTlmNGJlOWE0OWYxNDczNTlhODEwMzc=",  # NOQA: E501
}

PAGE_SIZE = 1000


class ElasicsearchClient:
    def __init__(self, environment: str, username: str, password: str):
        self.environment = environment
        self.username = username
        self.password = password
        self.index = f"ml-buoy-{environment}"
        self._cloud_id = _CLOUD_ID_MAP[environment]

        self._es = None

    @property
    def es(self):
        if self._es is None:
            self._es = Elasticsearch(
                cloud_id=self._cloud_id,
                basic_auth=(self.username, self.password),
                request_timeout=20,
                max_retries=3,
                retry_on_timeout=True,
            )

        return self._es

    def get_document(self, *, deployment_id: str, timestamp: int) -> Dict[str, Any]:
        try:
            document_id = f"{deployment_id}/{timestamp}"
            es_response = self.es.get(
                index=self.index,
                routing=deployment_id,
                id=document_id,
                source=METRICS,
            )

            document = es_response.body["_source"]
            for metric in document:
                document[metric] = document[metric][list(document[metric])[0]]

            return document

        except NotFoundError:
            return None

    def document_search(self, *, deployment_id: str, start: int, end: int):
        num_hits = None
        search_after = None

        sort = {"report_cycle_start_epoch.primary": {"order": "asc"}}

        query = {"bool": {"must": [{"term": {"_routing": {"value": deployment_id}}}]}}

        target_range_match = {
            "range": {
                "report_cycle_start_epoch.primary": {
                    "gte": start,
                    "lte": end,
                    # We need to specify the format or else Elasticsearch will assume
                    # the numbers are "epoch_millis" instead of "epoch_second"
                    "format": "epoch_second",
                },
            },
        }
        query["bool"]["must"].append(target_range_match)

        while num_hits is None or num_hits >= PAGE_SIZE:
            response = self.es.search(
                index=self.index,
                routing=deployment_id,
                sort=sort,
                size=PAGE_SIZE,
                search_after=search_after,
                query=query,
                source=METRICS,
            ).body

            hits = response["hits"]["hits"]
            for hit in hits:
                document = hit["_source"]
                for metric in document:
                    document[metric] = document[metric][list(document[metric])[0]]

                yield document

            num_hits = len(hits)
            if hits:
                search_after = hits[-1]["sort"]


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
    es_client = ElasicsearchClient(
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

    if json_output:
        filename = f"{filename}.json"
        output_file = output_directory / filename
        print(f"Writing output to: {output_file}")

        with output_file.open("w") as output:
            json.dump(es_documents, output, indent=2)

    else:
        filename = f"{filename}.csv"
        output_file = output_directory / filename
        print(f"Writing output to: {output_file}")

        with output_file.open("w") as output:
            writer = csv.DictWriter(
                output, [metric.split(".", 1)[0] for metric in METRICS]
            )
            writer.writeheader()
            writer.writerows(
                {
                    key: str(value) if value is not None else value
                    for key, value in es_document.items()
                }
                for es_document in es_documents
            )

        df = pd.read_csv(output_file)
        df = df.sort_values(["report_cycle_start_epoch"])
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Download time series buoy data for offline inspection. By default result "
            "data is returned in CSV format."
        )
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        choices=("production", "staging", "development"),
        default="production",
        help=(
            "The environment that you want to download images from "
            "(default: %(default)s)"
        ),
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
            "The ID of the buoy whose time series data you are downloading "
            "(e.g. prince_rupert_03)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).absolute().parent / "time_series_data",
        help=(
            "Specify a directory for the downloaded time series. Default is %(default)s"
        ),
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
            "The start and end time range (in Epoch seconds, inclusive) of time series "
            "data to download."
        ),
    )
    selection_group.add_argument(
        "--timestamp",
        type=int,
        help=(
            "The UNIX timestamp (seconds since Epoch) of the cycle "
            "(aka cycle_start_epoch) you are downloading data for."
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
