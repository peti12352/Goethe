"""ModelSpec detection (no CUDA)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ternary_runtime.profile import (
    ALL_SPECIALIZED_K,
    BONSAI,
    DEEPSEEK_V41,
    FLASH_NEXT,
    detect_from_hf,
    detect_from_pack,
    resolve_model,
)


def test_specialized_k_covers_public_models():
    assert set(ALL_SPECIALIZED_K) == {5120, 2304, 2560, 640}
    assert set(DEEPSEEK_V41.specialized_k) <= set(ALL_SPECIALIZED_K)
    assert set(FLASH_NEXT.specialized_k) <= set(ALL_SPECIALIZED_K)
    assert 17408 not in ALL_SPECIALIZED_K


def test_features_gate_models():
    assert DEEPSEEK_V41.has("moe_patch") and DEEPSEEK_V41.has("engram_readahead")
    assert DEEPSEEK_V41.has("fp4_fallback") and DEEPSEEK_V41.has("swiglu_clamp")
    assert not BONSAI.has("moe_patch")
    assert FLASH_NEXT.has("moe_patch") and FLASH_NEXT.has("bf16_fallback")
    assert not FLASH_NEXT.has("engram_readahead")


def test_resolve_forced(monkeypatch):
    monkeypatch.delenv("TERNARY_EXPERIMENTAL", raising=False)
    monkeypatch.setenv("TERNARY_PROFILE", "deepseek")
    monkeypatch.delenv("TERNARY_PACK_DIR", raising=False)
    assert resolve_model().name == "deepseek_v41"
    monkeypatch.setenv("TERNARY_PROFILE", "qwen4_exp")
    assert resolve_model().name == "flash_next"


def test_bonsai_profile_requires_experimental(monkeypatch):
    monkeypatch.delenv("TERNARY_EXPERIMENTAL", raising=False)
    monkeypatch.setenv("TERNARY_PROFILE", "bonsai")
    monkeypatch.delenv("TERNARY_PACK_DIR", raising=False)
    with pytest.raises(ValueError, match="unknown TERNARY_PROFILE"):
        resolve_model()
    monkeypatch.setenv("TERNARY_EXPERIMENTAL", "1")
    assert resolve_model().name == "bonsai"


def _env_pack(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    p = Path(raw)
    return p if (p / "layer-00.safetensors").is_file() else None


@pytest.mark.skipif(
    _env_pack("TERNARY_TEST_DEEPSEEK_PACK") is None,
    reason="set TERNARY_TEST_DEEPSEEK_PACK to a DeepSeek ternary pack dir",
)
def test_detect_deepseek_pack():
    p = detect_from_pack(str(_env_pack("TERNARY_TEST_DEEPSEEK_PACK")))
    assert p.name == "deepseek_v41"
    assert p.layout == "w123"


@pytest.mark.skipif(
    _env_pack("TERNARY_TEST_FLASH_NEXT_PACK") is None,
    reason="set TERNARY_TEST_FLASH_NEXT_PACK to a Flash-Next ternary pack dir",
)
def test_detect_flash_next_pack():
    p = detect_from_pack(str(_env_pack("TERNARY_TEST_FLASH_NEXT_PACK")))
    assert p.name == "flash_next"
    assert p.layout == "gate_up_down"


@pytest.mark.skipif(
    not os.environ.get("TERNARY_TEST_BONSAI_HF")
    or not Path(os.environ["TERNARY_TEST_BONSAI_HF"]).joinpath("config.json").is_file(),
    reason="set TERNARY_TEST_BONSAI_HF to a Bonsai HF checkpoint dir",
)
def test_detect_bonsai_hf(monkeypatch):
    monkeypatch.setenv("TERNARY_EXPERIMENTAL", "1")
    p = detect_from_hf(os.environ["TERNARY_TEST_BONSAI_HF"])
    assert p.name == "bonsai"
    assert p.kind == "dense"
    assert p.hidden == 5120 and p.intermediate == 17408
