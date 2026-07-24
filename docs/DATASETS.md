# Datasets

## VIGOR-M

The released dataset uses one portable root with no machine-specific paths:

```text
VIGOR-M/
├── README.md
├── manifest.json
├── panoramas/<city>/*.jpg
├── satellite/<city>/L0..L3/*.png
└── metadata/
    ├── city_bounds.csv
    ├── all.csv
    ├── same_area_train.csv
    ├── same_area_test.csv
    ├── cross_area_train.csv
    ├── cross_area_test.csv
    └── <city>/<city>_{all,train,test}.csv
```

The four city names are `Chicago`, `NewYork`, `SanFrancisco`, and `Seattle`.
Every `ground_path` in the released CSV files is relative to the dataset root.
The loader also recognizes the legacy local names `Pano/`, `level/`, and
`meta/level_pano/`, but new distributions should use the layout above.

### Preparing the publishable bundle

The reported same-area result is tied to a frozen split of 37,895 training and
37,789 test queries. The historical generator included an absolute local path in
its split hash, so regenerating the split after moving the data changes its
membership. Publish the frozen CSV files; do not ask users to regenerate them.

Stage a release from the raw assets and those frozen CSV files:

```bash
python scripts/data/prepare_vigor_m_release.py \
  --source-root /datasets/VIGOR-M-raw \
  --metadata-source /datasets/VIGOR-M-frozen-metadata \
  --output data/VIGOR-M \
  --mode symlink
```

`symlink` is intended for local validation without duplicating the images.
Use `--mode hardlink` for a same-filesystem staging area or `--mode copy` for a
self-contained directory that will be archived or uploaded. The command rewrites
only `ground_path`, preserves row order and split membership, validates every
referenced panorama/tile, checks the frozen split counts, and writes
`manifest.json` with metadata hashes and asset counts.

The resulting directory works directly with `--data-folder data/VIGOR-M`; a
separate `--metadata-folder` is unnecessary. Training uses dense L1 references at
stride 0.25 and native L2/L3 references.

`prepare_vigor_m_metadata.py` remains available only for custom panorama
collections. It now hashes stable `<city>/<filename>` keys and writes relative
paths to `metadata-generated/` by default, but its newly generated split is not
the frozen split behind the reported results.

## JustZoomIn

Download the official `pcvlab/justzoomin` Hugging Face dataset and follow its
archive extraction instructions without changing the repository structure. The
final directory placed at `data/justzoomin` is:

```text
justzoomin/
├── metadata/
│   ├── large_area_train_map.csv
│   └── large_area_val_map.csv
├── streetview/
│   └── images/<image_id>_undistorted.jpg
└── satellite/
    ├── layout.yaml
    └── {0,-1,-2,...,-9}/
```

The official package already supplies all metadata required by GeoMoE, including
the split CSV files, coordinates, and zoom-action sequences. No GeoMoE metadata
generation step is required.

Optionally precompute the derived dense L1/L2 satellite cache once:

```bash
python scripts/data/prepare_justzoomin_cache.py \
  --data-folder data/justzoomin \
  --levels L1 L2 \
  --splits train val \
  --satellite-stride-fraction 0.25
```

The default cache is under the ignored local `data/` area. It can be redirected
with `--cache-dir` and must not be committed or included in a source release.

## Licensing

The repository does not redistribute either dataset. Follow the dataset authors'
download terms and retain source attribution. JustZoomIn is distributed under CC
BY-SA 4.0 because it contains Mapillary-derived imagery; its Open Data DC aerial
component is CC BY 4.0.
