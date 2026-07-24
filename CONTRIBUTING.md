# Contributing

Keep changes scoped, format Python consistently, and run the release checks before
opening a pull request:

```bash
python -m compileall -q geomoe scripts tests
pytest -q
```

Do not commit datasets, feature caches, checkpoints, predictions, or the nested
`weights/` repository to the code repository. New reported results must state the
dataset split, checkpoint SHA256, beam width, calibration split, and candidate
cost together with accuracy.
