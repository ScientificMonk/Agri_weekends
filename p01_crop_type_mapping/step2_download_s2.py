"""Step 2: build monthly Sentinel-2 composites and extract a time series per field.

Output: data/processed/<region>/timeseries.npz with
  X       (fields, 12 months, 10 bands) mean reflectance per field
  y       crop class index per field
  classes class names
  x_m/y_m field centroids in metres (for spatial cross-validation)

Run from the repo root:  python -m p01_crop_type_mapping.step2_download_s2
Monthly composites are cached, so a re-run after a network error resumes.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize

from common import sentinel2 as s2

from . import config as C


def build_composites(region: str, fields: gpd.GeoDataFrame):
    cfg = C.REGIONS[region]
    cache = C.region_dir(region) / "composites"
    cache.mkdir(parents=True, exist_ok=True)

    # Small buffer so edge fields are fully covered
    bbox = list(fields.to_crs(4326).total_bounds)
    bbox = [bbox[0] - 0.01, bbox[1] - 0.01, bbox[2] + 0.01, bbox[3] + 0.01]

    catalog = s2.open_catalog()
    months = pd.date_range(cfg["season_start"], periods=12, freq="MS")
    grid_path = cache / "grid.json"
    for start in months:
        tag = start.strftime("%Y-%m")
        path = cache / f"{tag}.npy"
        if path.exists():
            print(f"{tag}: cached")
            continue
        end = (start + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        items = s2.search_items(catalog, bbox, start.strftime("%Y-%m-%d"), end, C.MAX_CLOUD)
        print(f"{tag}: {len(items)} scenes")
        result = s2.monthly_composite(items, bbox, cfg["crs"], C.RESOLUTION_M)
        if result is None:
            continue
        comp, geobox = result
        np.save(path, comp)
        if not grid_path.exists():
            grid_path.write_text(json.dumps({"transform": list(geobox.affine)[:6],
                                             "shape": list(geobox.shape)}))
        print(f"{tag}: clear pixels {np.isfinite(comp[0]).mean():.0%}")

    grid = json.loads(grid_path.read_text())
    return months, Affine(*grid["transform"]), tuple(grid["shape"]), cache


def field_means(comp: np.ndarray, ids: np.ndarray, n_fields: int) -> np.ndarray:
    """Mean of each band over each field's pixels, ignoring NaN. Returns (fields, bands)."""
    out = np.full((n_fields, comp.shape[0]), np.nan, dtype=np.float32)
    flat_ids = ids.ravel()
    for b in range(comp.shape[0]):
        vals = comp[b].ravel()
        ok = (flat_ids > 0) & np.isfinite(vals)
        sums = np.bincount(flat_ids[ok], vals[ok], minlength=n_fields + 1)
        cnts = np.bincount(flat_ids[ok], minlength=n_fields + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, b] = (sums / cnts)[1:]
    return out


def fill_gaps(X: np.ndarray) -> np.ndarray:
    """Linear interpolation over months that had no clear observation."""
    X = X.copy()
    t = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        for b in range(X.shape[2]):
            s = X[i, :, b]
            ok = np.isfinite(s)
            if ok.any() and not ok.all():
                X[i, :, b] = np.interp(t, t[ok], s[ok])
    return X


def main(region: str = C.REGION):
    out = C.region_dir(region)
    fields = gpd.read_file(out / "fields.gpkg")
    months, transform, shape, cache = build_composites(region, fields)

    # Label every pixel with the field it belongs to (0 = no field).
    # Small fields can miss every pixel centre, so a second pass uses all_touched.
    shapes = [(g, i + 1) for i, g in enumerate(fields.geometry)]
    ids = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    missing = set(range(1, len(fields) + 1)) - set(np.unique(ids))
    if missing:
        small = [(fields.geometry[i - 1], i) for i in missing]
        touched = rasterize(small, out_shape=shape, transform=transform, fill=0,
                            all_touched=True, dtype="int32")
        ids = np.where(ids == 0, touched, ids)

    X = np.full((len(fields), len(months), len(s2.BANDS)), np.nan, dtype=np.float32)
    for t, start in enumerate(months):
        path = cache / f"{start:%Y-%m}.npy"
        if path.exists():
            X[:, t] = field_means(np.load(path), ids, len(fields))

    observed = np.isfinite(X[:, :, 0]).sum(axis=1)
    keep = observed >= 6  # need at least half the season to say anything
    print(f"Fields with >=6 clear months: {keep.sum()} / {len(fields)}")
    X = fill_gaps(X[keep])
    fields = fields[keep].reset_index(drop=True)

    classes = np.array(sorted(fields.crop.unique()))
    y = np.searchsorted(classes, fields.crop.values)
    np.savez_compressed(
        out / "timeseries.npz", X=X, y=y, classes=classes,
        x_m=fields.x_m.values, y_m=fields.y_m.values,
        months=np.array([f"{m:%Y-%m}" for m in months]), bands=np.array(s2.BANDS),
    )
    print(f"Saved {X.shape} to {out / 'timeseries.npz'}")


if __name__ == "__main__":
    main()
