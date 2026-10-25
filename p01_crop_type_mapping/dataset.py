"""Load the field time series produced by step 2 and add vegetation indices."""
import numpy as np

from common.spatial_cv import block_ids

from . import config as C


def add_indices(X: np.ndarray, bands: list[str]) -> tuple[np.ndarray, list[str]]:
    """Append spectral indices as extra channels. X is (fields, months, bands)."""
    b = {name: X[..., i] for i, name in enumerate(bands)}
    eps = 1e-6
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    # EXERCISE 1: add more indices here and see if the baseline improves, e.g.
    #   NDRE  = (B8A - B05) / (B8A + B05)   red-edge, sensitive to chlorophyll
    #   NDMI  = (B08 - B11) / (B08 + B11)   canopy water content
    extra = [ndvi]
    names = ["NDVI"]
    return np.concatenate([X, np.stack(extra, axis=-1)], axis=-1), bands + names


def load(region: str = C.REGION):
    d = np.load(C.region_dir(region) / "timeseries.npz", allow_pickle=False)
    X, channels = add_indices(d["X"], list(d["bands"]))
    groups = block_ids(d["x_m"], d["y_m"], C.SPATIAL_BLOCK_M)
    return {
        "X": X.astype(np.float32),
        "y": d["y"],
        "classes": list(d["classes"]),
        "channels": channels,
        "months": list(d["months"]),
        "groups": groups,
    }
