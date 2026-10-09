"""SGLang integration: MoE substitution, checkpoint filter, readahead, post-load."""

from __future__ import annotations

import json
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generator, List, Optional, Tuple

import torch

from ternary_runtime.profile import PackProfile, active_profile

_PATCHED = False
_PROFILE: Optional[PackProfile] = None
_DS_EXPERT_KEY = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.\d+\.w[123]\.(weight|scale)$"
)
_QWEN_FUSED_EXPERT = re.compile(
    r"^(?:model\.language_model\.)?layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
_ENGRAM_EMBED = re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")


def apply_patches() -> None:
    """Install ternary hooks for the resolved model (MoE only; dense is a no-op)."""
    global _PATCHED, _PROFILE
    if _PATCHED:
        return
    pack_dir = os.environ.get("TERNARY_PACK_DIR", "").strip()
    try:
        _PROFILE = active_profile()
    except RuntimeError as exc:
        print(f"ternary: skip patch ({exc})", flush=True)
        return
    _PATCHED = True
    print(
        f"ternary: model={_PROFILE.name} kind={_PROFILE.kind} layout={_PROFILE.layout} "
        f"K*={_PROFILE.specialized_k} features={sorted(_PROFILE.features)}",
        flush=True,
    )
    if _PROFILE.kind != "moe" or not _PROFILE.has("moe_patch"):
        print(
            f"ternary: {_PROFILE.name} is dense; MoE/SGLang expert patches skipped",
            flush=True,
        )
        return
    if not pack_dir:
        print("ternary: MoE model needs TERNARY_PACK_DIR", flush=True)
        return
    print(
        f"ternary: patching SGLang layers=0..{_PROFILE.num_layers - 1} pack={pack_dir}",
        flush=True,
    )
    _patch_weight_iterators()
    _patch_moe_class(_PROFILE)
    _patch_model_runner_load()
    if _PROFILE.has("hc_mix_fast") and os.environ.get("TERNARY_HC_MIX", "1") != "0":
        _patch_hc_mix()
    if os.environ.get("TERNARY_SPEC_HOT_VOCAB", "").strip():
        _patch_spec_hot_vocab(os.environ["TERNARY_SPEC_HOT_VOCAB"].strip())


def _patch_spec_hot_vocab(path: str) -> None:
    """Replace the EAGLE/MTP drafter LM head with a replicated hot-token head.

    Hot rows live unevenly across vocab shards, so each rank contributes its rows (padded to a
    common length), one TP all-gather at init assembles the full [K, H] head on every rank, and
    the draft logits processor skips its per-step TP gather. hot_token_id maps draft indices
    back to the vocab; verification still uses the full target head, so outputs are unaffected.
    """
    from collections import OrderedDict

    import copy

    import sglang.srt.speculative.eagle_worker_v2 as ew

    orig = ew.EagleDraftWorker.init_lm_head

    def init_lm_head(self):
        orig(self)
        if self.hot_token_id is not None:
            return
        draft = self.draft_runner.model
        lm = draft.lm_head
        lp = getattr(draft, "logits_processor", None)
        tp_size = int(getattr(lm, "tp_size", 1))
        part = int(lm.num_embeddings_per_partition)
        vocab = int(lm.org_vocab_size)
        lo = int(lm.shard_indices.org_vocab_start_index)
        tp_rank = lo // part
        if lo != tp_rank * part or lm.weight.shape[0] != part or lp is None:
            print(f"ternary: hot vocab skipped (unexpected layout lo={lo})", flush=True)
            return
        hot = sorted({int(i) for i in torch.load(path, weights_only=True) if 0 <= int(i) < vocab})
        lists = [[i for i in hot if r * part <= i < (r + 1) * part] for r in range(tp_size)]
        width = max(len(x) for x in lists)
        dev = lm.weight.device
        w = lm.weight.data
        rows = torch.zeros(width, w.shape[1], dtype=w.dtype, device=dev)
        mine = lists[tp_rank]
        if mine:
            rows[: len(mine)] = w.index_select(0, torch.tensor(mine, device=dev) - lo)
        if tp_size > 1:
            from sglang.srt.distributed.parallel_state import get_tp_group

            rows = get_tp_group().all_gather(rows, dim=0)
        keep = torch.cat([torch.arange(len(x)) + r * width for r, x in enumerate(lists)]).to(dev)
        new_lm = copy.copy(lm)
        new_lm._parameters = OrderedDict(lm._parameters)
        new_lm.weight = torch.nn.Parameter(rows.index_select(0, keep).contiguous(), requires_grad=False)
        draft.lm_head = new_lm
        lp.do_tensor_parallel_all_gather = False
        lp.do_tensor_parallel_all_gather_dp_attn = False
        if hasattr(lp, "use_tp_lm_head_all_to_all"):
            lp.use_tp_lm_head_all_to_all = False
        if hasattr(lp, "_logits_gatherer"):
            lp._logits_gatherer.enabled = False
        self.hot_token_id = torch.tensor(hot, dtype=torch.int64, device=dev)
        print(
            f"ternary: spec hot vocab {len(hot)} ids -> replicated draft head {tuple(new_lm.weight.shape)} ({path})",
            flush=True,
        )

    ew.EagleDraftWorker.init_lm_head = init_lm_head


def _patch_hc_mix() -> None:
    """Route GatedResidual's decode-size HC mix to the split-K CUDA kernels."""
    import sglang.srt.layers.hyperconnection as hcmod

    orig = hcmod.fused_hc_mix

    def fused_hc_mix(hyper_input_normed, w_down, w_up, hc, hs):
        if (
            hc == 4
            and hyper_input_normed.shape[0] == 1
            and hyper_input_normed.dtype == torch.bfloat16
            and w_down.dtype == torch.bfloat16
            and w_up.dtype == torch.bfloat16
            and w_down.shape[0] % 64 == 0
        ):
            from ternary_runtime.kernels import hc_mix

            return hc_mix(hyper_input_normed, w_down, w_up, hc, hs)
        return orig(hyper_input_normed, w_down, w_up, hc, hs)

    hcmod.fused_hc_mix = fused_hc_mix
    print("ternary: hc_mix -> flash_decode split-K kernels", flush=True)


def _skip_routed_expert_key(name: str, profile: Optional[PackProfile] = None) -> bool:
    """Skip checkpoint tensors replaced by ternary packs for this profile."""
    profile = profile or _PROFILE
    if profile is None or profile.kind != "moe":
        return False
    if profile.layout == "w123":
        m = _DS_EXPERT_KEY.match(name)
        return m is not None and profile.is_ternary_layer(int(m.group(1)))
    if profile.layout == "gate_up_down":
        m = _QWEN_FUSED_EXPERT.match(name)
        return m is not None and profile.is_ternary_layer(int(m.group(1)))
    return False


def _patch_moe_class(profile: PackProfile) -> None:
    import sglang.srt.layers.moe.ep_moe.layer as ep_layer

    orig = ep_layer.get_moe_impl_class
    n_layers = profile.num_layers

    def get_moe_impl_class(quant_config):
        stock_cls = orig(quant_config)

        class MoESelector:
            def __call__(self, *args, **kwargs):
                layer_id = kwargs.get("layer_id", -1)
                prefix = str(kwargs.get("prefix", ""))
                # MTP / nextn keep published experts when profile requests it.
                if profile.has("mtp_skip_ternary") and (
                    "mtp" in prefix or "nextn" in prefix or "is_nextn" in kwargs
                ):
                    return stock_cls(*args, **kwargs)
                if 0 <= layer_id < n_layers:
                    from ternary_runtime.experts import TernaryExperts

                    return TernaryExperts(*args, **kwargs)
                return stock_cls(*args, **kwargs)

        return MoESelector()

    ep_layer.get_moe_impl_class = get_moe_impl_class


def _parse_file_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header


def _tensor_data_offset(header: dict, key: str, header_size: int) -> Tuple[int, int, tuple, str]:
    meta = header[key]
    begin, end = meta["data_offsets"]
    return 8 + header_size + begin, end - begin, tuple(meta["shape"]), meta["dtype"]


def _engram_byte_range(
    shape: tuple,
    dtype: str,
    tp_rank: int,
    tp_size: int,
    file_offset: int,
) -> Tuple[int, int]:
    rows = shape[0]
    bits = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "U8": 1, "F8_E4M3": 1, "F8_E8M0": 1}
    row_bytes = bits.get(dtype, 1)
    for dim in shape[1:]:
        row_bytes *= dim
    row_start = rows * tp_rank // tp_size
    row_end = rows * (tp_rank + 1) // tp_size
    off = file_offset + row_start * row_bytes
    length = (row_end - row_start) * row_bytes
    return off, length


def _prefault(path: str, offset: int, length: int) -> None:
    if length <= 0:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
        except (AttributeError, OSError):
            pass
        chunk = 1 << 22
        pos = offset
        end = offset + length
        while pos < end:
            n = min(chunk, end - pos)
            os.pread(fd, n, pos)
            pos += n
    finally:
        os.close(fd)


class _ReadaheadPlan:
    """Per-rank byte ranges to prefault inside checkpoint shards."""

    def __init__(self, weight_files: List[str], weight_map: Dict[str, str]):
        self._shard_paths: Dict[str, str] = {}
        for wf in weight_files:
            self._shard_paths[os.path.basename(wf)] = wf
        self._headers: Dict[str, dict] = {}
        self._header_size: Dict[str, int] = {}
        for base, abspath in self._shard_paths.items():
            with open(abspath, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._header_size[base] = n
                self._headers[base] = json.loads(f.read(n))
        self._keys_by_file: Dict[str, List[str]] = {b: [] for b in self._shard_paths}
        for key, shard in weight_map.items():
            if _skip_routed_expert_key(key):
                continue
            base = os.path.basename(shard)
            if base in self._keys_by_file:
                self._keys_by_file[base].append(key)

    def ranges_for_file(self, shard_base: str, tp_rank: int, tp_size: int) -> List[Tuple[int, int]]:
        header = self._headers[shard_base]
        hsize = self._header_size[shard_base]
        abspath = self._shard_paths[shard_base]
        base_off = 8 + hsize
        out: List[Tuple[int, int]] = []
        for key in self._keys_by_file.get(shard_base, []):
            if key not in header:
                continue
            meta = header[key]
            begin, end = meta["data_offsets"]
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            if (
                _PROFILE is not None
                and _PROFILE.has("engram_readahead")
                and _ENGRAM_EMBED.match(key)
                and tp_size > 1
            ):
                off, length = _engram_byte_range(
                    shape, dtype, tp_rank, tp_size, base_off + begin
                )
            else:
                off, length = base_off + begin, end - begin
            out.append((off, length))
        return out


class _ReadaheadPool:
    def __init__(self, plan: _ReadaheadPlan, workers: int, tp_rank: int, tp_size: int):
        self._plan = plan
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._pending = {}
        self._lock = threading.Lock()
        self._files = list(plan._shard_paths.keys())

    def schedule(self, file_index: int) -> None:
        ahead = 3
        with self._lock:
            for j in range(1, ahead + 1):
                idx = file_index + j
                if idx >= len(self._files):
                    break
                base = self._files[idx]
                if base in self._pending:
                    continue
                ranges = self._plan.ranges_for_file(base, self._tp_rank, self._tp_size)
                path = self._plan._shard_paths[base]

                def _run(p=path, rs=ranges):
                    for off, ln in rs:
                        _prefault(p, off, ln)

                self._pending[base] = self._pool.submit(_run)

    def wait_file(self, shard_base: str) -> None:
        with self._lock:
            fut = self._pending.pop(shard_base, None)
        if fut is not None:
            fut.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def _keys_for_file(weight_map: Dict[str, str], shard_path: str) -> List[str]:
    base = os.path.basename(shard_path)
    return sorted(k for k, v in weight_map.items() if os.path.basename(v) == base and not _skip_routed_expert_key(k))


def _ternary_safetensors_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Safetensors iterator that skips routed expert tensors covered by ternary packs."""
    import itertools
    from tqdm.auto import tqdm

    import safetensors.torch
    from safetensors import safe_open

    import sglang.srt.model_loader.weight_utils as wu
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import BAR_FORMAT

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    tp_rank = get_parallel().tp_rank if torch.distributed.is_initialized() else 0
    tp_size = get_parallel().tp_size if torch.distributed.is_initialized() else 1
    workers = int(os.environ.get("TERNARY_LOAD_WORKERS", "16"))

    ckpt_dir = os.path.dirname(hf_weights_files[0]) if hf_weights_files else ""
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"] if os.path.isfile(index_path) else {}

    plan = _ReadaheadPlan(hf_weights_files, weight_map)
    pool = _ReadaheadPool(plan, workers, tp_rank, tp_size)
    shard_list = list(plan._shard_paths.keys())

    if prefetch and not disable_mmap:
        wu._prefetch_all_checkpoints(sorted(hf_weights_files), num_threads=prefetch_num_threads)

    for file_idx, st_file in enumerate(
        tqdm(
            hf_weights_files,
            desc="Loading safetensors checkpoint shards",
            disable=not enable_tqdm,
            bar_format=BAR_FORMAT,
            position=tqdm._get_free_pos(),
        )
    ):
        shard_base = os.path.basename(st_file)
        pool.schedule(file_idx)
        pool.wait_file(shard_base)
        keys = _keys_for_file(weight_map, st_file) if weight_map else None
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
            names = sorted(k for k in result if not _skip_routed_expert_key(k))
            for name in names:
                yield name, result[name]
        else:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                names = keys if keys is not None else [
                    k for k in f.keys() if not _skip_routed_expert_key(k)
                ]
                for name in names:
                    yield name, f.get_tensor(name)
        if drop_cache_after_load:
            wu._drop_file_cache_after_load(st_file)

    pool.shutdown()


def _ternary_buffered_iterator(
    hf_weights_files: List[str],
    max_workers: int,
    disable_mmap: bool = False,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Multi-threaded loader that never materializes skipped expert tensors."""
    import collections
    import concurrent.futures
    import itertools

    from tqdm.auto import tqdm
    from safetensors import safe_open

    import safetensors.torch
    import sglang.srt.model_loader.weight_utils as wu
    from sglang.srt.utils import BAR_FORMAT

    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    ckpt_dir = os.path.dirname(hf_weights_files[0]) if hf_weights_files else ""
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"] if os.path.isfile(index_path) else {}

    if prefetch and not disable_mmap:
        wu._prefetch_all_checkpoints(sorted(hf_weights_files), num_threads=prefetch_num_threads)

    def _load_file(st_file: str):
        keys = _keys_for_file(weight_map, st_file) if weight_map else None
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
            return {k: result[k] for k in (keys or result) if k in result}
        with safe_open(st_file, framework="pt", device="cpu") as f:
            names = keys if keys is not None else [k for k in f.keys() if not _skip_routed_expert_key(k)]
            return {n: f.get_tensor(n) for n in names}

    buffer_size = max_workers + 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_iter = iter(hf_weights_files)
        pending: collections.deque = collections.deque()
        for st_file in itertools.islice(file_iter, buffer_size):
            pending.append((st_file, executor.submit(_load_file, st_file)))
        with tqdm(
            total=len(hf_weights_files),
            desc="Ternary multi-thread loading shards",
            disable=not enable_tqdm,
            bar_format=BAR_FORMAT,
            position=tqdm._get_free_pos(),
        ) as pbar:
            while pending:
                st_file, future = pending.popleft()
                state_dict = future.result()
                del future
                next_file = next(file_iter, None)
                if next_file is not None:
                    pending.append((next_file, executor.submit(_load_file, next_file)))
                for name in sorted(state_dict.keys()):
                    yield name, state_dict[name]
                del state_dict
                if drop_cache_after_load:
                    wu._drop_file_cache_after_load(st_file)
                pbar.update(1)


def _patch_weight_iterators() -> None:
    import sglang.srt.model_loader.weight_utils as wu

    wu.safetensors_weights_iterator = _ternary_safetensors_iterator
    wu.buffered_multi_thread_safetensors_weights_iterator = _ternary_buffered_iterator
    # Rebind aliases captured by `from weight_utils import ...` if loader is already imported.
    try:
        import sglang.srt.model_loader.loader as loader_mod

        loader_mod.safetensors_weights_iterator = _ternary_safetensors_iterator
        loader_mod.buffered_multi_thread_safetensors_weights_iterator = (
            _ternary_buffered_iterator
        )
    except Exception:
        pass


def _patch_model_runner_load() -> None:
    from sglang.srt.model_executor.model_runner import ModelRunner

    orig = ModelRunner.load_model

    def load_model(self):
        orig(self)
        _load_ternary_modules(self.model)

    ModelRunner.load_model = load_model


def _load_ternary_modules(model: torch.nn.Module) -> None:
    pack_dir = os.environ.get("TERNARY_PACK_DIR", "").strip()
    if not pack_dir:
        return
    from ternary_runtime.experts import TernaryExperts

    modules = [m for m in model.modules() if isinstance(m, TernaryExperts)]
    if not modules:
        print("ternary: no TernaryExperts modules found", flush=True)
        return
    try:
        from sglang.srt.runtime_context import get_model

        ckpt_dir = get_model().model_path
    except Exception:
        ckpt_dir = os.environ.get("MODEL_PATH", "")
    if not ckpt_dir:
        raise RuntimeError("ternary load: cannot resolve checkpoint path")
    device = torch.device("cuda", torch.cuda.current_device())
    TernaryExperts.load_all(modules, pack_dir, ckpt_dir, device)
