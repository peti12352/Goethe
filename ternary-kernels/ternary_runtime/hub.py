"""Download ternary packs + base checkpoints for plug-and-play use.

A Hugging Face ternary repo ships the expert pack and ``ternary_config.json``.
The base model is fetched from the original HF id listed in that config (no need
to re-upload hundreds of GB of base weights into the ternary repo).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_NAME = "ternary_config.json"


@dataclass(frozen=True)
class ResolvedModel:
    """Local paths after resolve/download."""

    pack_dir: Path
    base_dir: Path
    profile: str
    config: Dict[str, Any]
    repo_id: Optional[str] = None

    def apply_env(self) -> None:
        """Export env vars consumed by sitecustomize / bootstrap."""
        os.environ["TERNARY_PACK_DIR"] = str(self.pack_dir)
        os.environ["TERNARY_PROFILE"] = self.profile
        os.environ["TERNARY_MODEL_DIR"] = str(self.base_dir)
        os.environ.setdefault("MODEL_PATH", str(self.base_dir))


def _cache_root() -> Path:
    override = os.environ.get("TERNARY_HF_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "ternary-flash" / "hub"
    return Path.home() / ".cache" / "ternary-flash" / "hub"


def load_ternary_config(pack_dir: Path) -> Dict[str, Any]:
    path = Path(pack_dir) / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {CONFIG_NAME} under {pack_dir}; "
            "expected a ternary HF pack or prepared local pack"
        )
    return json.loads(path.read_text())


def _snapshot(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    allow_patterns: Optional[list] = None,
    local_dir: Optional[Path] = None,
) -> Path:
    from huggingface_hub import snapshot_download

    kwargs: Dict[str, Any] = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
        kwargs["local_dir_use_symlinks"] = False
    path = snapshot_download(**kwargs)
    return Path(path)


def fetch(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    base_local: Optional[str] = None,
    apply_env: bool = True,
) -> ResolvedModel:
    """Download ternary pack from ``repo_id`` and its declared base model.

    Parameters
    ----------
    repo_id:
        HF repo containing layer-*.safetensors + ternary_config.json
        (e.g. ``org/qwen38-flash-next-ternary-latest``).
    base_local:
        Optional existing local base checkpoint; skips base download when set.
    """
    root = Path(cache_dir) if cache_dir else _cache_root()
    pack_local = root / "packs" / repo_id.replace("/", "__")
    pack_dir = _snapshot(
        repo_id,
        revision=revision,
        local_dir=pack_local,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "README.md",
            "LICENSE*",
            "*.yaml",
            "*.yml",
            "*.txt",
        ],
    )
    cfg = load_ternary_config(pack_dir)
    profile = cfg.get("profile") or cfg.get("ternary_profile")
    if not profile:
        raise ValueError(f"{CONFIG_NAME} missing profile")
    base_id = cfg.get("base_model")
    if not base_id and not base_local:
        raise ValueError(f"{CONFIG_NAME} missing base_model")
    base_rev = cfg.get("base_revision")
    if base_local:
        base_dir = Path(base_local)
        if not base_dir.is_dir():
            raise FileNotFoundError(base_dir)
    else:
        base_cache = root / "bases" / str(base_id).replace("/", "__")
        base_dir = _snapshot(str(base_id), revision=base_rev, local_dir=base_cache)
    resolved = ResolvedModel(
        pack_dir=pack_dir,
        base_dir=base_dir,
        profile=str(profile),
        config=cfg,
        repo_id=repo_id,
    )
    if apply_env:
        resolved.apply_env()
    return resolved


def resolve_local(
    pack_dir: str | Path,
    *,
    base_dir: Optional[str | Path] = None,
    apply_env: bool = True,
) -> ResolvedModel:
    """Resolve a local pack that already has ternary_config.json."""
    pack = Path(pack_dir)
    cfg = load_ternary_config(pack)
    profile = cfg.get("profile") or cfg.get("ternary_profile")
    if not profile:
        raise ValueError(f"{CONFIG_NAME} missing profile")
    if base_dir is not None:
        base = Path(base_dir)
    elif cfg.get("base_model_local"):
        base = Path(cfg["base_model_local"])
    else:
        raise ValueError("pass base_dir= or set base_model_local in ternary_config.json")
    resolved = ResolvedModel(
        pack_dir=pack,
        base_dir=base,
        profile=str(profile),
        config=cfg,
        repo_id=cfg.get("hf_repo"),
    )
    if apply_env:
        resolved.apply_env()
    return resolved
