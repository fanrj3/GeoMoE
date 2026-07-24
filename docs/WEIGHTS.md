# Weight Repository

`weights/` is an independent Git repository nested inside this code checkout.
The parent `.gitignore` excludes it, so GitHub contains no model binaries. Clone
the Hugging Face repository directly at that path:

The Hugging Face clone URL will be added after the first public model upload.
Clone that repository to `weights/` in the code checkout.

Expected layout:

```text
weights/
├── manifest.json
├── vigor_m/
│   ├── geomoe_b11_e5_e60.pth
│   ├── path_calibrator.pt
│   └── metrics.json
└── justzoomin/
    ├── geomoe_b11_e5_e60.pth
    ├── path_calibrator.pt
    └── metrics.json
```

Verify sizes and SHA256 hashes before evaluation:

```bash
python scripts/tools/verify_weights.py --weights weights
```

Git LFS is required before committing or pushing the binary repository:

```bash
git lfs install
git -C weights add .gitattributes manifest.json vigor_m justzoomin
```
