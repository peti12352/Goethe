import importlib.util
import os

_pack = os.environ.get("TERNARY_PACK_DIR", "").strip()
_profile = os.environ.get("TERNARY_PROFILE", "").strip()
_model = os.environ.get("TERNARY_MODEL_DIR", "").strip() or os.environ.get(
    "MODEL_PATH", ""
).strip()
if (_pack or _profile or _model) and importlib.util.find_spec("sglang"):
    from ternary_runtime import bootstrap

    bootstrap()
