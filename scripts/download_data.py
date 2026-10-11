"""
Dataset acquisition for the Crop Advisory Assistant.

Resolution order (first match wins):
1. A ready-made merged CSV at DATA_CSV_PATH (data/raw/crop_yield.csv).
2. The three files of the Kaggle "Crop Yield Data with Soil and Weather" dataset
   (crop_yield.csv, state_soil_data.csv, state_weather_data_*.csv) found in
   KAGGLE_SOURCE_DIR (default ./data_set), data/raw/kaggle or data/raw. They are
   merged into DATA_CSV_PATH.
3. A Kaggle API download (needs KAGGLE_USERNAME / KAGGLE_KEY), then merged as in (2).
4. A clearly labelled synthetic dataset (data/raw/SYNTHETIC_crop_yield.csv).
"""
import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from scripts.generate_synthetic_data import generate_synthetic_crop_dataset

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "anshumish/crop-yield-data-with-soil-and-weather"


def find_kaggle_files(directory: Path) -> Optional[Tuple[Path, Path, Path]]:
    """Returns (yield_csv, soil_csv, weather_csv) if all three Kaggle files exist in `directory`."""
    if not directory.is_dir():
        return None
    yield_csv = directory / "crop_yield.csv"
    soil_csv = directory / "state_soil_data.csv"
    weather_matches = sorted(glob.glob(str(directory / "state_weather_data*.csv")))
    if yield_csv.exists() and soil_csv.exists() and weather_matches:
        return yield_csv, soil_csv, Path(weather_matches[0])
    return None


def merge_kaggle_files(yield_csv: Path, soil_csv: Path, weather_csv: Path, out_path: Path) -> Path:
    """
    Joins the Kaggle yield table with state-level soil (by state) and state-year weather
    (by state + year) and writes one flat CSV.

    The raw `fertilizer` and `pesticide` columns are totals in kg for the whole cultivated
    area, so they are divided by `area` to obtain kg/ha (see docs/ASSUMPTIONS.md).
    """
    crops = pd.read_csv(yield_csv)
    soil = pd.read_csv(soil_csv, encoding="utf-8-sig")
    weather = pd.read_csv(weather_csv)

    for frame in (crops, soil, weather):
        frame.columns = [c.strip() for c in frame.columns]
        frame["state"] = frame["state"].astype(str).str.strip()

    year_col = "year" if "year" in crops.columns else "crop_year"
    merged = crops.merge(soil, on="state", how="left")
    merged = merged.merge(weather, left_on=["state", year_col], right_on=["state", "year"], how="left",
                          suffixes=("", "_weather"))
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_weather")])

    area = merged["area"].where(merged["area"] > 0)
    merged["fertilizer"] = (merged["fertilizer"] / area).round(2)
    merged["pesticide"] = (merged["pesticide"] / area).round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    missing = int(merged[["N", "avg_temp_c"]].isna().any(axis=1).sum())
    print(f"Merged Kaggle files -> {out_path} ({len(merged)} rows, {missing} rows missing soil/weather)")
    return out_path


def _candidate_source_dirs() -> list[Path]:
    raw_dir = Path(settings.DATA_CSV_PATH).parent
    return [Path(settings.KAGGLE_SOURCE_DIR), raw_dir / "kaggle", raw_dir]


def real_dataset_available() -> bool:
    """True if a merged real CSV or the three Kaggle source files can be found locally."""
    raw_path = Path(settings.DATA_CSV_PATH)
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        return True
    return any(find_kaggle_files(d) for d in _candidate_source_dirs())


def ensure_dataset() -> str:
    """
    Returns the path of the raw CSV to ingest, producing it if necessary (see module docstring).
    """
    raw_path = Path(settings.DATA_CSV_PATH)
    synth_path = Path(settings.SYNTHETIC_DATA_PATH)

    if raw_path.exists() and raw_path.stat().st_size > 1000:
        print(f"Found existing real dataset at: {raw_path}")
        return str(raw_path)

    for directory in _candidate_source_dirs():
        files = find_kaggle_files(directory)
        if files and files[0].resolve() != raw_path.resolve():
            print(f"Found Kaggle source files in {directory}; merging...")
            return str(merge_kaggle_files(*files, out_path=raw_path))

    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if kaggle_user and kaggle_key:
        try:
            print("Attempting to download dataset from Kaggle...")
            import kaggle
            kaggle.api.authenticate()
            download_dir = raw_path.parent / "kaggle"
            download_dir.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(download_dir), unzip=True)
            files = find_kaggle_files(download_dir)
            if files:
                return str(merge_kaggle_files(*files, out_path=raw_path))
            print(f"Kaggle download finished but expected files were not found in {download_dir}")
        except Exception as e:
            print(f"Kaggle download failed or kaggle package not configured: {e}")

    print("=" * 78)
    print("WARNING: No real dataset found. Generating a SYNTHETIC dataset for demo purposes.")
    print(f"Place the Kaggle files in {settings.KAGGLE_SOURCE_DIR} or a merged CSV at {raw_path}.")
    print("=" * 78)
    generate_synthetic_crop_dataset(str(synth_path), num_rows=6000)
    return str(synth_path)


def write_data_profile(raw_path: str, out_path: Path) -> None:
    """Writes docs/DATA_PROFILE.md from the raw CSV and the cleaned dataset."""
    from backend.app.data.ingest import get_active_dataframe, get_dataset_info
    from backend.app.data.features import load_thresholds

    raw = pd.read_csv(raw_path)
    clean = get_active_dataframe()
    info = get_dataset_info()
    thresholds = load_thresholds(settings.THRESHOLDS_PATH)
    numeric = clean.select_dtypes("number").drop(columns=["yield_percentile_within_crop"], errors="ignore")

    lines = [
        "# Data Profile",
        "",
        "_Generated by `python scripts/download_data.py`; re-run it after changing the dataset._",
        "",
        f"- **Source file**: `{Path(raw_path).relative_to(BASE_DIR) if Path(raw_path).is_relative_to(BASE_DIR) else raw_path}`",
        f"- **Synthetic**: {'YES - demo data only' if info.get('is_synthetic') else 'no (Kaggle anshumish crop yield + state soil + state weather)'}",
        f"- **Raw rows**: {len(raw):,}  |  **Cleaned rows**: {len(clean):,}",
        f"- **Crops**: {clean['crop'].nunique()}  |  **States**: {clean['state'].nunique()}  |  "
        f"**Seasons**: {', '.join(sorted(clean['season'].unique()))}  |  "
        f"**Years**: {clean['crop_year'].min()}-{clean['crop_year'].max()}",
        "",
        "## Raw columns (as found in the CSV)",
        "",
        "| Column | dtype | Nulls | Example |",
        "|---|---|---|---|",
    ]
    for col in raw.columns:
        example = raw[col].dropna().iloc[0] if raw[col].notna().any() else ""
        lines.append(f"| `{col}` | {raw[col].dtype} | {int(raw[col].isna().sum())} | {str(example).strip()} |")

    lines += ["", "## Cleaned numeric columns (canonical names)", "",
              "| Column | mean | std | min | 25% | 50% | 75% | max |", "|---|---|---|---|---|---|---|---|"]
    desc = numeric.describe().T
    for col, row in desc.iterrows():
        lines.append(f"| `{col}` | " + " | ".join(f"{row[k]:,.2f}" for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]) + " |")

    lines += ["", "## Rows per crop (top 15)", "", "| Crop | Rows | Median yield |", "|---|---|---|"]
    by_crop = clean.groupby("crop")["yield"].agg(["size", "median"]).sort_values("size", ascending=False).head(15)
    for crop, row in by_crop.iterrows():
        lines.append(f"| {crop} | {int(row['size'])} | {row['median']:.2f} |")

    lines += ["", "## Derived class thresholds (`data/processed/thresholds.json`)", "", "```json",
              json.dumps(thresholds, indent=2), "```", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote data profile -> {out_path}")


if __name__ == "__main__":
    import argparse
    from backend.app.data.ingest import clean_and_prepare_dataset

    parser = argparse.ArgumentParser(description="Resolve, clean and profile the crop dataset.")
    parser.add_argument("--no-profile", action="store_true", help="Skip writing docs/DATA_PROFILE.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    path = ensure_dataset()
    clean_and_prepare_dataset(path)
    if not args.no_profile:
        write_data_profile(path, BASE_DIR / "docs" / "DATA_PROFILE.md")
