# Multicam Image Processing

A repo to capture code/scripts related to processing multicam images

## Scripts

### Requirements

Before running any of the scripts, make sure to have the latest Python requirements install.
You can install the latest requirements by running the following command from the root directory of this repo:

```bash
pip install -r requirements.txt
```

### download_images.py

The [download_images.py](scripts/download_images.py) script can be used to easily downloading images from a given buoy.

#### Usage

To use the `download_images.py` script, first switch to the `scripts/` directory.

For detailed usage instructions run the following from the `scripts/` directory:

```bash
python download_images.py --help
```

To fetch data for a single timestamp, the following example command can be used:
```bash
python download_images.py -d prince_rupert_03 --timestamp 1725642300
```

To fetch data for a range of timestamps, the following example command can be used:
```bash
python download_images.py -d prince_rupert_03 --time-range 1725642300 1725643200
```

To fetch data for a list of timestamp defined in a JSON file, the following example command can be used:
```bash
python download_images.py -d prince_rupert_03 --json-file epoch_list.json
```

### download_timeseries_data.py

The [download_timeseries_data.py](scripts/download_timeseries_data.py) script can be used to easily downloading time series data for a given buoy in CSV or JSON format.
```

#### Usage

To use the `download_timeseries_data.py` script, first switch to the `scripts/` directory.

For detailed usage instructions run the following from the `scripts/` directory:

```bash
./download_timeseries_data.py --help
```

**NOTE:** The following examples assume that you have the `ES_USERNAME` and `ES_PASSWORD` environment variables set with your Elasticsearch credentials.

To fetch data for a single timestamp in CSV format, the following example command can be used:
```bash
./download_timeseries_data.py -u $ES_USERNAME -p $ES_PASSWORD -d prince_rupert_03 --timestamp 1725995100
```

To fetch data for a range of timestamps in CSV format, the following example command can be used:
```bash
./download_timeseries_data.py -u $ES_USERNAME -p $ES_PASSWORD -d prince_rupert_03 --time-range 1704067200 1717200000
```

You can also use the `--json` flag to output the returned time series data in JSON format. e.g.,
```bash
./download_timeseries_data.py -u $ES_USERNAME -p $ES_PASSWORD -d prince_rupert_03 --timestamp 1725995100 --json