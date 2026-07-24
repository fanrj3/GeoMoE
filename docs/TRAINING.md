# Training

Both trainers launch one process per selected GPU internally; do not wrap them in
`torchrun`. The released checkpoints use DINOv2 ViT-B/14, a routed FFN at the
zero-based Block 11, five experts, Top-2 routing, and 60 epochs. Query/reference
InfoNCE is evaluated separately per scale, preventing cross-scale negatives.

## VIGOR-M

```bash
python scripts/train/train_vigor_m.py \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/meta/level_pano \
  --gpu-ids 0 \
  --master-port 12362 \
  --run-name vigor_m_b11_e5
```

The final configuration uses L1/L2/L3, dense L1 stride 0.25, batch size 128 per
GPU, learning rate `1e-4`, DINOv2 initialization, GPS and visual hard-negative
sampling, and level-wise InfoNCE.

## JustZoomIn

```bash
python scripts/train/train_justzoomin.py \
  --data-folder data/justzoomin \
  --satellite-cache-dir data/justzoomin/.cache/geomoe_satellite \
  --gpu-ids 0 \
  --master-port 12358 \
  --run-name justzoomin_b11_e5
```

This setup trains L1-L4 jointly, uses dense stride-0.25 references at L1/L2,
weights the L4 loss by 2, and otherwise matches the shared B11/E5 settings.

## Resume and Alternative Initialization

Pass `--init-mode checkpoint --checkpoint-start <path>` to initialize from a
GeoMoE checkpoint. `level_ckpts` initialization is retained for ablations and
requires one checkpoint per level via `--l1-checkpoint`, etc. The paper models
use `--init-mode pretrained`.

Checkpoints, logs, the exact trainer snapshot, and the model definition are saved
below `outputs/checkpoints/`.
