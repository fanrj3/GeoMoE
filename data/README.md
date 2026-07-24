# Local Data Mounts

Datasets and generated caches are intentionally ignored by the code repository.
The examples assume the canonical, directly usable layouts:

```text
data/
├── VIGOR-M/
│   ├── panoramas/<city>/
│   ├── satellite/<city>/L0..L3/
│   ├── metadata/city_bounds.csv
│   └── metadata/*.csv            # frozen release splits, not regenerated
└── justzoomin/
    ├── metadata/large_area_{train,val}_map.csv
    ├── streetview/images/
    ├── satellite/layout.yaml
    └── satellite/{0,-1,...,-9}/
```

Symlinks are supported, for example:

```bash
ln -s /datasets/VIGOR-M data/VIGOR-M
ln -s /datasets/justzoomin data/justzoomin
```

The official JustZoomIn download already contains its required metadata. Build a
canonical VIGOR-M bundle with `scripts/data/prepare_vigor_m_release.py`; see
`docs/DATASETS.md` for the release and validation procedure.
