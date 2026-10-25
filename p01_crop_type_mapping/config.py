"""Paths and region settings for Project 1."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS_URL = "https://zenodo.org/records/7257688/files/labels.geojson?download=1"
LABELS_PATH = DATA / "raw" / "labels.geojson"
RESULTS = Path(__file__).resolve().parent / "results"

# Each region is one CropHarvest dataset. CropHarvest labels describe the 12
# months *before* export_end_date, so `season_start` is that date minus a year.
REGIONS = {
    "germany": {
        "dataset": "germany",
        "season_start": "2018-02-01",
        "crs": "EPSG:32633",  # UTM 33N
    },
    # Exercise: once Germany works, try a bigger and harder region
    "ile-de-france": {
        "dataset": "ile-de-france",
        "season_start": "2019-02-01",
        "crs": "EPSG:32631",  # UTM 31N
    },
}

REGION = "germany"
RESOLUTION_M = 20
MAX_CLOUD = 70           # skip scenes that are mostly cloud
MIN_CLASS_SIZE = 50      # rarer classes are merged into "other"
SPATIAL_BLOCK_M = 2000   # block size for spatial cross-validation
SEED = 42


def region_dir(region: str = REGION) -> Path:
    return DATA / "processed" / region
