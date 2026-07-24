# Datasets

## VIGOR-M

GeoMoE expects four cities (`Chicago`, `NewYork`, `SanFrancisco`, and `Seattle`),
panoramas under `Pano/<city>/`, level tiles under `level/<city>/L*`, and the city
bounds file at `figures/pano_distribution/pano_distribution_summary.csv`.

Generate the deterministic same-area and cross-area CSV files after placing the
dataset:

```bash
python scripts/data/prepare_vigor_m_metadata.py \
  --root data/VIGOR-M \
  --output data/VIGOR-M/meta/level_pano
```

The released main result uses the same-area protocol and all 37,789 official test
queries. Training uses dense L1 references at stride 0.25 and native L2/L3
references.

## JustZoomIn

Download and extract the official `pcvlab/justzoomin` Hugging Face dataset. The
loader uses `metadata/large_area_train_map.csv` and
`metadata/large_area_val_map.csv`, street-view images, and the multi-resolution
satellite hierarchy described by `satellite/layout.yaml`.

Precompute the dense L1/L2 satellite crops once:

```bash
python scripts/data/prepare_justzoomin_cache.py \
  --data-folder data/justzoomin \
  --levels L1 L2 \
  --splits train val \
  --satellite-stride-fraction 0.25
```

The cache is derived data and should remain outside Git.

## Licensing

The repository does not redistribute either dataset. Follow the dataset authors'
download terms and retain source attribution. JustZoomIn is distributed under CC
BY-SA 4.0 because it contains Mapillary-derived imagery; its Open Data DC aerial
component is CC BY 4.0.
