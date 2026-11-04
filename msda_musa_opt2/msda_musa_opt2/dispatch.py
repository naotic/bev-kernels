from __future__ import annotations

from typing import Tuple

import torch


def _ext():
    # Local import to avoid import-time hard failure on non-MUSA env.
    import _msda_opt2_musa  # type: ignore

    return _msda_opt2_musa


def ms_deform_attn_forward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc: torch.Tensor,
    attn_weight: torch.Tensor,
    *,
    im2col_step: int = 64,
    force_generic: bool = False,
) -> torch.Tensor:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("msda_musa_opt2 requires torch.musa.is_available()")
    return _ext().forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step=im2col_step,
        force_generic=force_generic,
    )


def ms_deform_attn_backward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc: torch.Tensor,
    attn_weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    im2col_step: int = 64,
    force_generic_col2im: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("msda_musa_opt2 requires torch.musa.is_available()")

    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_loc)
    grad_attn_weight = torch.zeros_like(attn_weight)

    _ext().backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        grad_value,
        grad_sampling_loc,
        grad_attn_weight,
        im2col_step=im2col_step,
        force_generic_col2im=force_generic_col2im,
    )
    return grad_value, grad_sampling_loc, grad_attn_weight

