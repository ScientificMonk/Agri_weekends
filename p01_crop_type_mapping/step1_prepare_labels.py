"""Step 1: download CropHarvest labels and keep the crop-type fields for one region.

Run from the repo root:  python -m p01_crop_type_mapping.step1_prepare_labels
"""
import geopandas as gpd
import requests

from . import config as C


def download_labels():
    if C.LABELS_PATH.exists():
        return
    C.LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading CropHarvest labels (~70 MB)...")
    with requests.get(C.LABELS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(C.LABELS_PATH, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def main(region: str = C.REGION):
    download_labels()
    cfg = C.REGIONS[region]
    labels = gpd.read_file(C.LABELS_PATH)
    fields = labels[(labels.dataset == cfg["dataset"]) & labels.classification_label.notna()].copy()

    # Merge classes too rare to learn into "other"
    counts = fields.classification_label.value_counts()
    rare = counts[counts < C.MIN_CLASS_SIZE].index
    fields["crop"] = fields.classification_label.where(~fields.classification_label.isin(rare), "other")

    fields = fields[["crop", "geometry"]].reset_index(drop=True)
    fields["field_id"] = fields.index
    fields = fields.to_crs(cfg["crs"])
    centroids = fields.geometry.centroid
    fields["x_m"], fields["y_m"] = centroids.x, centroids.y

    out = C.region_dir(region)
    out.mkdir(parents=True, exist_ok=True)
    fields.to_file(out / "fields.gpkg", driver="GPKG")
    print(f"{region}: {len(fields)} fields saved to {out / 'fields.gpkg'}")
    print(fields.crop.value_counts().to_string())
    if len(rare):
        print(f"Merged into 'other': {list(rare)}")


if __name__ == "__main__":
    main()
