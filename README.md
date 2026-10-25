# Agri Weekends

Weekend-sized machine learning and deep learning projects on agriculture, soil health and climate, built on free satellite and climate data. Each project lives in its own folder, reuses the shared code in [`common/`](common), and reports results with **spatial cross-validation**, so the numbers hold up on new areas and not just near the training fields.

## Projects

| # | Project | Status | Best result |
|---|---|---|---|
| 01 | [Crop type mapping from Sentinel-2 time series](p01_crop_type_mapping) | Baseline done, deep models in progress | RF macro-F1 0.809 (spatial CV) |
| 02 | Soil organic carbon prediction (LUCAS + Sentinel-2 + terrain) | Planned | |
| 03 | Drought and crop stress early warning (NDVI and soil moisture forecasting) | Planned | |
| 04 | Fine-tuning a geospatial foundation model (Prithvi / Clay) | Planned | |
| 05 | Leaf disease detection that holds up on field photos | Planned | |
| 06 | Cloud removal for Sentinel-2 | Planned | |
| 07 | Climate-to-yield explainability | Planned | |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or pick a CUDA build at pytorch.org
```

Satellite data comes from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/), which needs no account. Run everything from the repo root, for example `python -m p01_crop_type_mapping.step1_prepare_labels`.

## Repo layout

```
common/                 shared code: Sentinel-2 composites, spatial CV
p01_crop_type_mapping/  one folder per project, with its own README and results/
data/                   downloaded and processed data (git-ignored, re-creatable)
```
