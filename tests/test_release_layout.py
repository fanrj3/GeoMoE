"""Static smoke tests for the source and optional nested weights release."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_entrypoints_exist():
    """Keep all documented train/eval/data commands in the release bundle."""
    required = (
        "scripts/train/train_vigor_m.py",
        "scripts/train/train_justzoomin.py",
        "scripts/eval/prepare_vigor_m_features.py",
        "scripts/eval/evaluate_vigor_m.py",
        "scripts/eval/evaluate_justzoomin.py",
        "scripts/data/prepare_vigor_m_release.py",
        "scripts/data/prepare_vigor_m_metadata.py",
        "scripts/data/prepare_justzoomin_cache.py",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_all_python_files_parse():
    """Catch syntax errors in every shipped Python module and entrypoint."""
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_method_defaults_are_locked_to_b11_e5():
    """Protect the published B11/E5/top-2 defaults from accidental drift."""
    for relative in (
        "scripts/train/train_vigor_m.py",
        "scripts/train/train_justzoomin.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "moe_start_block: int = 11" in source
        assert "num_experts: int = 5" in source
        assert "top_k: int = 2" in source
        assert 'init_mode: str = "pretrained"' in source


def test_weight_manifest_when_present():
    """Check the nested weight manifest schema without requiring weight files."""
    manifest_path = ROOT / "weights/manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "geomoe-weights-manifest-v1"
    assert len(payload["artifacts"]) == 6
