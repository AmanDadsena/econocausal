"""Download the Online Retail II dataset from the UCI repository.

Usage:
    python -m econocausal.data.download
    python -m econocausal.data.download --force   # download again
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from econocausal.paths import RAW_DIR

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_NAME = "online_retail_ii.zip"
EXCEL_NAME = "online_retail_II.xlsx"


def download_file(url: str, dest: Path) -> Path:
    """Download `url` to `dest`. Writes to a temp file first so a failed
    download never leaves a half-written file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest


def extract_excel(zip_path: Path, out_dir: Path) -> Path:
    """Extract the Excel file from the downloaded zip and return its path."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid zip file, try --force to download again")

    with zipfile.ZipFile(zip_path) as zf:
        excel_files = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not excel_files:
            raise ValueError(f"No .xlsx file found in {zip_path}")
        member = excel_files[0]
        target = out_dir / EXCEL_NAME
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def download_dataset(force: bool = False) -> Path:
    """Download and extract the dataset into data/raw. Skips the download
    if the Excel file already exists, unless `force` is True."""
    excel_path = RAW_DIR / EXCEL_NAME
    if excel_path.exists() and not force:
        print(f"Already downloaded: {excel_path}")
        return excel_path

    zip_path = RAW_DIR / ZIP_NAME
    print(f"Downloading {DATASET_URL}")
    download_file(DATASET_URL, zip_path)
    print(f"Saved zip ({zip_path.stat().st_size / 1e6:.1f} MB)")

    excel_path = extract_excel(zip_path, RAW_DIR)
    zip_path.unlink()
    print(f"Extracted {excel_path}")
    return excel_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Online Retail II dataset.")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
