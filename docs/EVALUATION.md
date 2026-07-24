# Evaluation

## VIGOR-M: Fixed K=4 + PRC

First extract normalized train and test features with the same backbone:

```bash
python scripts/eval/prepare_vigor_m_features.py \
  --split train --checkpoint weights/vigor_m/geomoe_b11_e5_e60.pth \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/metadata \
  --output outputs/cache/vigor_m_train.pt

python scripts/eval/prepare_vigor_m_features.py \
  --split test --checkpoint weights/vigor_m/geomoe_b11_e5_e60.pth \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/metadata \
  --output outputs/cache/vigor_m_test.pt
```

To reproduce the released result with the provided train-only PRC:

```bash
python scripts/eval/evaluate_vigor_m.py \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/metadata \
  build-table \
  --bundle outputs/cache/vigor_m_test.pt \
  --calibrator weights/vigor_m/path_calibrator.pt \
  --output outputs/cache/vigor_m_test_prc_table.pt

python scripts/eval/evaluate_vigor_m.py \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/metadata \
  evaluate-current \
  --calibrated-test-table outputs/cache/vigor_m_test_prc_table.pt \
  --output outputs/eval/vigor_m_current_method.json
```

To refit PRC without test supervision, build an uncalibrated train table and run:

```bash
python scripts/eval/evaluate_vigor_m.py \
  --data-folder data/VIGOR-M \
  --metadata-folder data/VIGOR-M/metadata \
  build-table \
  --bundle outputs/cache/vigor_m_train.pt \
  --output outputs/cache/vigor_m_train_table.pt

python scripts/eval/evaluate_vigor_m.py train-calibrator \
  --table outputs/cache/vigor_m_train_table.pt \
  --output outputs/checkpoints/vigor_m_path_calibrator.pt
```

The table builder retains all K1/K2 actions from 1 through 8 for ablations, but
the released method always reports K1=K2=4.

## JustZoomIn: Fixed K=4 + PRC

The JustZoomIn evaluator performs extraction, fixed-beam search, and reporting in
one entry point. Passing the released calibrator skips fitting and preserves the
official train-to-val protocol:

```bash
python scripts/eval/evaluate_justzoomin.py \
  --checkpoint weights/justzoomin/geomoe_b11_e5_e60.pth \
  --calibrator-checkpoint weights/justzoomin/path_calibrator.pt \
  --data-folder data/justzoomin \
  --output-dir outputs/eval/justzoomin_current_method
```

Omit `--calibrator-checkpoint` to refit PRC on the training split. The evaluator
also emits the exhaustive flat L4 diagnostic. That row is not the hierarchical
main method and must not be reported as Beam + PRC.
