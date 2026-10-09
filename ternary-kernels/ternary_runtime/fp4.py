"""Unpack DeepSeek FP4 E2M1 expert weights.

Two values per byte. The low nibble is the earlier element along K.
One E8M0 scale covers 32 unpacked K values.
"""

import torch

# OCP E2M1 magnitudes. Bit 3 is the sign.
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
FP4_GROUP = 32


def e2m1_table(device=None) -> torch.Tensor:
    return torch.tensor(E2M1, dtype=torch.float32, device=device)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """packed: integer tensor, two FP4 values per byte, shape [..., K/2]."""
    raw = packed.view(torch.uint8)
    table = e2m1_table(raw.device)
    lo = table[(raw & 0x0F).long()]
    hi = table[(raw >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = unpack_fp4(packed)
    scales = scale.to(torch.float32)
    if values.shape[-1] != scales.shape[-1] * FP4_GROUP:
        raise ValueError(
            f"K {values.shape[-1]} is not {FP4_GROUP} times scale groups {scales.shape[-1]}"
        )
    return values * scales.repeat_interleave(FP4_GROUP, dim=-1)
