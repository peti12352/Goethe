"""ternary_config.json contract (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ternary_runtime.hub import CONFIG_NAME, load_ternary_config, resolve_local


def test_load_ternary_config_roundtrip(tmp_path: Path):
    cfg = {
        "format": "ternary-g128-hadamard-gptq",
        "profile": "flash_next",
        "base_model": "Qwen/Qwen3.8-Flash-Next",
        "base_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "pack_layout": "gate_up_down",
        "n_layers": 48,
    }
    (tmp_path / CONFIG_NAME).write_text(json.dumps(cfg))
    loaded = load_ternary_config(tmp_path)
    assert loaded["profile"] == "flash_next"
    base = tmp_path / "base"
    base.mkdir()
    resolved = resolve_local(tmp_path, base_dir=base, apply_env=False)
    assert resolved.profile == "flash_next"
    assert resolved.base_dir == base


def test_missing_config_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_ternary_config(tmp_path)
