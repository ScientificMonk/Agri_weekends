"""Sentinel-2 L2A access via Microsoft Planetary Computer (no account needed).

Builds cloud-masked monthly median composites for a bounding box, which is the
workhorse for most projects in this repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
from odc.stac import load

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# 10 m and 20 m spectral bands (the 60 m atmospheric bands are skipped)
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Scene Classification Layer values we treat as clear sky:
# 4 vegetation, 5 bare soil, 6 water, 7 unclassified, 11 snow
CLEAR_SCL = [4, 5, 6, 7, 11]


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_items(catalog, bbox, start: str, end: str, max_cloud: float = 70):
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def _boa_offset(item) -> int:
    # From processing baseline 04.00 (Jan 2022) reflectances carry a +1000 offset
    baseline = item.properties.get("s2:processing_baseline", "00.00")
    return 1000 if float(baseline) >= 4.0 else 0


def monthly_composite(items, bbox, crs: str, resolution: int = 20):
    """Cloud-masked median of all scenes in `items`.

    Returns (composite, geobox): float32 reflectance in [0, 1] shaped
    (bands, y, x) with NaN where no clear observation exists, and the pixel
    grid it sits on. Returns None if there were no scenes.
    """
    if not items:
        return None
    ds = load(
        items,
        bands=BANDS + ["SCL"],
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        dtype="uint16",
        nodata=0,
    )
    clear = ds["SCL"].isin(CLEAR_SCL).values  # (time, y, x)

    # groupby="solar_day" merges tiles; take the offset from any item on that day
    day_offset = {}
    for it in items:
        day_offset[pd.Timestamp(it.datetime).date()] = _boa_offset(it)
    offsets = np.array([day_offset.get(pd.Timestamp(t).date(), 0) for t in ds.time.values])

    out = np.empty((len(BANDS),) + clear.shape[1:], dtype=np.float32)
    for i, band in enumerate(BANDS):
        arr = ds[band].values.astype(np.float32)
        valid = clear & (arr > 0)
        arr = (arr - offsets[:, None, None]) / 10000.0
        arr[~valid] = np.nan
        with np.errstate(all="ignore"):
            out[i] = np.nanmedian(arr, axis=0)
    return out, ds.odc.geobox
