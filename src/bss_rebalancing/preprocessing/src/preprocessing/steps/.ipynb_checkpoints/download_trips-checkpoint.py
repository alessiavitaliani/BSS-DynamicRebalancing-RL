"""
Download trip data from BlueBikes (Cambridge) and Citi Bike NYC (Manhattan).

This module downloads and extracts the trip data for the specified year.
"""

import argparse
import os
import shutil
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from preprocessing.config import PreprocessingConfig, DEFAULT_CONFIG


def download_and_extract(url: str, target_directory: str, tbar: tqdm = None) -> None:
    """
    Download a file from the given URL and extract if it's a ZIP file.

    Parameters:
        url: The URL to download the file from.
        target_directory: The directory where the file will be saved and extracted.
        tbar: Optional tqdm progress bar for status updates.
    """
    try:
        os.makedirs(target_directory, exist_ok=True)

        filename = os.path.basename(urlparse(url).path)
        save_path = os.path.join(target_directory, filename)

        if tbar is not None:
            tbar.set_description(f"Downloading {filename}")
        else:
            print(f"Downloading file from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        if zipfile.is_zipfile(save_path):
            if tbar is not None:
                tbar.set_description(f"Extracting {filename}")
            else:
                print(f"Extracting contents of {save_path}...")

            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(target_directory)

            os.remove(save_path)
            if tbar is not None:
                tbar.set_description(f"Removed ZIP: {filename}")
            else:
                print(f"Removed the ZIP file: {save_path}")
        else:
            print(f"The file is not a ZIP archive. No extraction performed.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def run(config: PreprocessingConfig) -> None:
    """
    Run the download trips step.

    Parameters:
        config: The preprocessing configuration.
    """
    save_path = config.trips_path
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving trip data to: {save_path}")

    tbar = tqdm(config.months, desc="Downloading files", position=0, leave=True)

    for month in config.months:
        #filename = f"{config.year}{str(month).zfill(2)}-bluebikes-tripdata.csv" # Cambridge
        month_str = str(month).zfill(2)
        filename = f"{config.year}{month_str}-citibike-tripdata.csv" # Manhattan
        if not os.path.exists(os.path.join(save_path, filename)):
            #url = f"https://s3.amazonaws.com/hubway-data/{config.year}{str(month).zfill(2)}-bluebikes-tripdata.zip" # Cambridge
            url = url = f"https://s3.amazonaws.com/tripdata/{config.year}{month_str}-citibike-tripdata.zip" # Manhattan
            download_and_extract(url, save_path, tbar)
        else:
            tbar.set_description(f"Skipping {filename} (exists)")
        tbar.update(1)

    # Clean up macOS artifacts
    macosx_path = os.path.join(save_path, "__MACOSX")
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)


def main():
    """CLI entry point for download_trips."""
    parser = argparse.ArgumentParser(description="Download the BlueBikes trip data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CONFIG.data_path,
        help="The directory where the data will be saved.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_CONFIG.year,
        help="Year of data to download.",
    )

    args = parser.parse_args()

    config = PreprocessingConfig(data_path=args.data_path, year=args.year)
    run(config)


if __name__ == "__main__":
    main()
