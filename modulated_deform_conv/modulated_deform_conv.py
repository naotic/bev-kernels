"""
Modulated Deformable Conv2d: CUDA/MUSA extension wrapper + pure PyTorch golden implementation.

The golden implementation uses differentiable PyTorch ops so that col2im / col2im_coord
can be obtained via torch.autograd.grad on the im2col forward.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
import os

# ---------------------------------------------------------------------------
# Extension wrapper (thin layer over the compiled C++ module)
# ---------------------------------------------------------------------------

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        import _modulated_deform_conv_ext as ext
        _ext = ext
    return _ext


def ext_modulated_deformable_im2col(
    data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    batch_size, channels, height, width = data_im.shape
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    data_col = data_im.new_zeros(channels * kernel_h * kernel_w, batch_size, height_col, width_col)
    ext.modulated_deformable_im2col_cuda(
        data_im, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, data_col,
    )
    return data_col


def ext_modulated_deformable_im2col_opt(
    data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    """Baseline im2col for optimization experiments (same API, simple implementation)."""
    ext = _load_ext()
    batch_size, channels, height, width = data_im.shape
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    data_col = data_im.new_zeros(channels * kernel_h * kernel_w, batch_size, height_col, width_col)
    ext.modulated_deformable_im2col_opt_cuda(
        data_im, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, data_col,
    )
    return data_col


def ext_modulated_deformable_col2im_opt(
    data_col, data_offset, data_mask,
    batch_size, channels, height, width,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    grad_im = data_col.new_zeros(batch_size, channels, height, width)
    ext.modulated_deformable_col2im_opt_cuda(
        data_col, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, grad_im,
    )
    return grad_im


def ext_modulated_deformable_col2im_smem(
    data_col, data_offset, data_mask,
    batch_size, channels, height, width,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    grad_im = data_col.new_zeros(batch_size, channels, height, width)
    ext.modulated_deformable_col2im_smem_cuda(
        data_col, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, grad_im,
    )
    return grad_im


def ext_modulated_deformable_col2im(
    data_col, data_offset, data_mask,
    batch_size, channels, height, width,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    grad_im = data_col.new_zeros(batch_size, channels, height, width)
    ext.modulated_deformable_col2im_cuda(
        data_col, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, grad_im,
    )
    return grad_im


def ext_modulated_deformable_col2im_coord(
    data_col, data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    batch_size, channels, height, width = data_im.shape
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    grad_offset = data_offset.new_zeros(data_offset.shape)
    grad_mask = data_mask.new_zeros(data_mask.shape)
    ext.modulated_deformable_col2im_coord_cuda(
        data_col, data_im, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, grad_offset, grad_mask,
    )
    return grad_offset, grad_mask


def ext_modulated_deformable_col2im_coord_opt(
    data_col, data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    ext = _load_ext()
    batch_size, channels, height, width = data_im.shape
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    grad_offset = data_offset.new_zeros(data_offset.shape)
    grad_mask = data_mask.new_zeros(data_mask.shape)
    ext.modulated_deformable_col2im_coord_opt_cuda(
        data_col, data_im, data_offset, data_mask,
        batch_size, channels, height, width, height_col, width_col,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group, grad_offset, grad_mask,
    )
    return grad_offset, grad_mask


# ---------------------------------------------------------------------------
# Pure PyTorch golden implementation (differentiable)
# ---------------------------------------------------------------------------

def _bilinear_sample(data_im_ptr, height, width, h, w):
    """
    Differentiable bilinear sampling matching the CUDA kernel boundary behavior.
    data_im_ptr: [H, W]  (single channel, single batch)
    h, w: [N] sampling coordinates (float)
    Returns: [N] sampled values
    """
    h_low = h.floor().long()
    w_low = w.floor().long()
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low.float()
    lw = w - w_low.float()
    hh = 1.0 - lh
    hw = 1.0 - lw

    def _get(iy, ix):
        valid = (iy >= 0) & (iy <= height - 1) & (ix >= 0) & (ix <= width - 1)
        iy_c = iy.clamp(0, height - 1)
        ix_c = ix.clamp(0, width - 1)
        val = data_im_ptr[iy_c, ix_c]
        return torch.where(valid, val, torch.zeros_like(val))

    v1 = _get(h_low, w_low)
    v2 = _get(h_low, w_high)
    v3 = _get(h_high, w_low)
    v4 = _get(h_high, w_high)

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4


def modulated_deformable_im2col_pytorch(
    data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    """
    Pure PyTorch im2col. All inputs must be contiguous float tensors.
    data_im:     [B, C, H, W]
    data_offset: [B, deform_group * 2 * kH * kW, H_out, W_out]
    data_mask:   [B, deform_group * kH * kW, H_out, W_out]
    Returns data_col: [C * kH * kW, B, H_out, W_out]
    """
    batch_size, num_channels, height, width = data_im.shape
    height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    channel_per_deformable_group = num_channels // deformable_group

    h_col = torch.arange(height_col, device=data_im.device, dtype=data_im.dtype)
    w_col = torch.arange(width_col, device=data_im.device, dtype=data_im.dtype)
    h_grid, w_grid = torch.meshgrid(h_col, w_col, indexing='ij')
    h_grid = h_grid.reshape(-1)
    w_grid = w_grid.reshape(-1)

    data_col_list = []
    for c_im in range(num_channels):
        dg_idx = c_im // channel_per_deformable_group
        for i in range(kernel_h):
            for j in range(kernel_w):
                for b in range(batch_size):
                    h_in = h_grid * stride_h - pad_h
                    w_in = w_grid * stride_w - pad_w

                    offset_h = data_offset[b, dg_idx * 2 * kernel_h * kernel_w + 2 * (i * kernel_w + j), :, :].reshape(-1)
                    offset_w = data_offset[b, dg_idx * 2 * kernel_h * kernel_w + 2 * (i * kernel_w + j) + 1, :, :].reshape(-1)
                    mask_val = data_mask[b, dg_idx * kernel_h * kernel_w + i * kernel_w + j, :, :].reshape(-1)

                    h_im = h_in + i * dilation_h + offset_h
                    w_im = w_in + j * dilation_w + offset_w

                    valid = (h_im > -1) & (w_im > -1) & (h_im < height) & (w_im < width)
                    h_im_c = torch.where(valid, h_im, torch.zeros_like(h_im))
                    w_im_c = torch.where(valid, w_im, torch.zeros_like(w_im))

                    val = _bilinear_sample(data_im[b, c_im], height, width, h_im_c, w_im_c)
                    val = torch.where(valid, val, torch.zeros_like(val))
                    data_col_list.append((val * mask_val).reshape(height_col, width_col))

    c_kk = num_channels * kernel_h * kernel_w
    data_col = torch.stack(data_col_list, dim=0).reshape(c_kk, batch_size, height_col, width_col)
    return data_col


def modulated_deformable_col2im_pytorch(
    data_col, data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    """
    Compute grad_im via autograd on the differentiable im2col forward.
    data_col: [C*kH*kW, B, H_out, W_out] -- treated as grad_output (upstream gradient)
    Returns grad_im: [B, C, H, W]
    """
    data_im_var = data_im.detach().clone().requires_grad_(True)
    col = modulated_deformable_im2col_pytorch(
        data_im_var, data_offset.detach(), data_mask.detach(),
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group,
    )
    grad_im = torch.autograd.grad(col, data_im_var, grad_outputs=data_col, retain_graph=False)[0]
    return grad_im


def modulated_deformable_col2im_coord_pytorch(
    data_col, data_im, data_offset, data_mask,
    kernel_h, kernel_w, pad_h, pad_w,
    stride_h, stride_w, dilation_h, dilation_w,
    deformable_group,
):
    """
    Compute grad_offset and grad_mask via autograd on the differentiable im2col forward.
    data_col: [C*kH*kW, B, H_out, W_out] -- treated as grad_output (upstream gradient)
    Returns (grad_offset, grad_mask)
    """
    offset_var = data_offset.detach().clone().requires_grad_(True)
    mask_var = data_mask.detach().clone().requires_grad_(True)
    col = modulated_deformable_im2col_pytorch(
        data_im.detach(), offset_var, mask_var,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        deformable_group,
    )
    grads = torch.autograd.grad(col, [offset_var, mask_var], grad_outputs=data_col, retain_graph=False)
    return grads[0], grads[1]


# ---------------------------------------------------------------------------
# Complete Modulated Deform Conv2d (im2col + GEMM), equivalent to mmcv
# ---------------------------------------------------------------------------

class ModulatedDeformConv2dFunction(Function):
    """Drop-in replacement for mmcv.ops.ModulatedDeformConv2dFunction.

    Batched im2col/col2im in single kernel launches.  GEMM uses mm directly
    on the channel-major columns layout produced by the kernel, eliminating
    all permute + contiguous copies on the large column matrix.

    Key insight: im2col outputs columns as [C*kH*kW, B*H*W] (channel-major).
    When groups=1 (the common case for DCN), weight @ columns is a single mm.
    When groups>1, we loop over the (small) number of groups with per-group mm.
    The only remaining transpose is on the (much smaller) output tensor.
    """

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None,
                stride=1, padding=0, dilation=1, groups=1, deform_groups=1):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')

        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)

        input = input.type_as(offset)
        weight = weight.type_as(input)
        ctx.save_for_backward(input, offset, mask, weight, bias)

        batch = input.size(0)
        channels = input.size(1)
        height = input.size(2)
        width = input.size(3)
        channels_out = weight.size(0)
        c_out_g = channels_out // groups
        c_in_g_kkw = (channels // groups) * weight.size(2) * weight.size(3)

        kernel_h, kernel_w = weight.size(2), weight.size(3)
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation

        height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        hw_out = height_out * width_out
        bhw = batch * hw_out

        # im2col: full batch, output layout [C*kH*kW, B*H_out*W_out]
        columns = input.new_zeros(channels * kernel_h * kernel_w, bhw)

        ext = _load_ext()
        ext.modulated_deformable_im2col_opt_cuda(
            input, offset, mask,
            batch, channels, height, width, height_out, width_out,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deform_groups, columns)

        # GEMM: weight[G, c_out_g, c_in_g_kkw] @ columns[G, c_in_g_kkw, B*hw]
        # columns is already [C*kH*kW, B*hw] contiguous, just view as [G, ...]
        # Each group slice is a contiguous [c_in_g_kkw, bhw] block — no copy.
        col_g = columns.view(groups, c_in_g_kkw, bhw)
        weight_g = weight.view(groups, c_out_g, c_in_g_kkw)

        # output_g: [G, c_out_g, B*hw]
        if groups == 1:
            output_flat = torch.mm(weight_g[0], col_g[0]).unsqueeze(0)
        else:
            output_flat = torch.empty(groups, c_out_g, bhw,
                                      dtype=columns.dtype, device=columns.device)
            for g in range(groups):
                torch.mm(weight_g[g], col_g[g], out=output_flat[g])

        # Reshape [G, c_out_g, B*hw] → [C_out, B, H, W] → [B, C_out, H, W]
        # This is the only transpose, on the output tensor (much smaller than columns)
        output = output_flat.view(channels_out, batch, height_out, width_out) \
                            .permute(1, 0, 2, 3) \
                            .contiguous()

        if ctx.with_bias:
            output = output + bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)

        grad_output = grad_output.contiguous()

        batch = input.size(0)
        channels = input.size(1)
        height = input.size(2)
        width = input.size(3)
        channels_out = weight.size(0)
        groups = ctx.groups
        deform_groups = ctx.deform_groups
        c_out_g = channels_out // groups
        c_in_g_kkw = (channels // groups) * weight.size(2) * weight.size(3)

        kernel_h, kernel_w = weight.size(2), weight.size(3)
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation

        height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        hw_out = height_out * width_out
        bhw = batch * hw_out

        ext = _load_ext()

        # grad_output: [B, C_out, H, W] → [C_out, B*hw] (channel-major)
        # This mirrors the columns layout so weight^T @ grad_out produces
        # columns in the right [C*kH*kW, B*hw] layout directly.
        grad_out_cm = grad_output.permute(1, 0, 2, 3).contiguous().view(channels_out, bhw)
        grad_out_g = grad_out_cm.view(groups, c_out_g, bhw)

        weight_g = weight.view(groups, c_out_g, c_in_g_kkw)

        # ---- Step 1: columns = weight^T @ grad_output ----
        # weight_g^T: [G, c_in_g_kkw, c_out_g] @ grad_out_g: [G, c_out_g, B*hw]
        # → columns: [G, c_in_g_kkw, B*hw] → view [C*kH*kW, B*hw] (contiguous, no copy)
        columns = input.new_empty(groups, c_in_g_kkw, bhw)
        if groups == 1:
            torch.mm(weight_g[0].t(), grad_out_g[0], out=columns[0])
        else:
            for g in range(groups):
                torch.mm(weight_g[g].t(), grad_out_g[g], out=columns[g])
        columns = columns.view(channels * kernel_h * kernel_w, bhw)

        # ---- Step 2: grad_offset, grad_mask (full batch) ----
        ext.modulated_deformable_col2im_coord_opt_cuda(
            columns, input, offset, mask,
            batch, channels, height, width, height_out, width_out,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deform_groups,
            grad_offset, grad_mask)

        # ---- Step 3: grad_input (full batch) ----
        ext.modulated_deformable_col2im_smem_cuda(
            columns, offset, mask,
            batch, channels, height, width, height_out, width_out,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deform_groups, grad_input)

        # ---- Step 4: im2col again for grad_weight (full batch) ----
        ext.modulated_deformable_im2col_opt_cuda(
            input, offset, mask,
            batch, channels, height, width, height_out, width_out,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deform_groups, columns)

        # grad_weight[g] = grad_out_g[g] @ columns_g[g]^T
        # columns: [C*kH*kW, B*hw] → [G, c_in_g_kkw, B*hw]
        col_g = columns.view(groups, c_in_g_kkw, bhw)

        # grad_out_g[g]: [c_out_g, B*hw] @ col_g[g]^T: [B*hw, c_in_g_kkw]
        # → [c_out_g, c_in_g_kkw]  (batch sum is implicit in the mm)
        grad_weight = weight.new_empty(groups, c_out_g, c_in_g_kkw)
        if groups == 1:
            torch.mm(grad_out_g[0], col_g[0].t(), out=grad_weight[0])
        else:
            for g in range(groups):
                torch.mm(grad_out_g[g], col_g[g].t(), out=grad_weight[g])
        grad_weight = grad_weight.view_as(weight)

        # ---- Step 5: grad_bias ----
        if ctx.with_bias:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        else:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):
    """Drop-in replacement for mmcv.ops.ModulatedDeformConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """Self-contained DCNv2 module that generates offsets and masks internally.

    Drop-in replacement for mmcv.ops.ModulatedDeformConv2dPack.
    Weights are fully compatible with mmcv checkpoints.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        # backend selection, priority: explicit kwarg > DCN_BACKEND env > 'cuda'.
        #  'cuda'    : existing torchada/MUSA ext im2col path (default; lowest mem).
        #  'fuse_c'  : native NHWC fused (muDNN offset conv + custom gather + muBLAS
        #              GEMM). Default. With DCN_RECOMPUTE_COLS=1 (recompute im2col
        #              columns in backward, default on) it fits full BEVFormer at
        #              samples_per_gpu=2 (50.3GB) and is ~2% faster end-to-end than
        #              'cuda' (3.44 vs 3.51 s/iter). groups==1 & deform_groups==1
        #              only (else auto-falls back to 'cuda').
        #  'cuda'    : existing torchada/MUSA ext im2col path (fallback).
        #  'tilelang': fused-tl path.
        self.backend = kwargs.pop('backend', os.environ.get('DCN_BACKEND', 'fuse_c'))
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        backend = getattr(self, 'backend', 'fuse_c')
        # fuse_c / tilelang only support groups==1 & deform_groups==1; otherwise
        # fall back to the general 'cuda' im2col path.
        if backend in ('fuse_c', 'tilelang') and (
                self.groups != 1 or self.deform_groups != 1):
            backend = 'cuda'
        if backend == 'tilelang':
            # tilelang backend: conv_offset is computed inside (TME c2d_im2col), so we pass
            # its weight/bias straight through; grads flow back to conv_offset.{weight,bias}.
            assert self.groups == 1 and self.deform_groups == 1, \
                "tilelang backend supports groups==1, deform_groups==1 only"
            from .fused_tl import dcn_apply_nchw
            return dcn_apply_nchw(x, self.conv_offset.weight, self.conv_offset.bias,
                                  self.weight, self.bias,
                                  self.stride, self.padding, self.dilation)
        if backend == 'fuse_c':
            # hand-written NHWC CUDA fused path (muDNN offset conv + custom gather +
            # muBLAS GEMM fwd; per-element col2im bwd). conv_offset computed inside;
            # grads flow back to conv_offset.{weight,bias}. NCHW-in/out adapter.
            assert self.groups == 1 and self.deform_groups == 1, \
                "fuse_c backend supports groups==1, deform_groups==1 only"
            from .dcn_split_nhwc import dcn_apply_nchw as dcn_apply_nchw_fusec
            return dcn_apply_nchw_fusec(x, self.conv_offset.weight, self.conv_offset.bias,
                                        self.weight, self.bias,
                                        self.stride, self.padding, self.dilation)
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        if os.environ.get("ENABLE_CHANNEL_LAST", "false").lower() in ("1", "true", "yes", "on"):
            x=x.to(memory_format=torch.contiguous_format)
            offset=offset.to(memory_format=torch.contiguous_format)
            mask=mask.to(memory_format=torch.contiguous_format)
            return modulated_deform_conv2d(x, offset, mask, self.weight.to(memory_format=torch.contiguous_format), self.bias,
                                           self.stride, self.padding,
                                           self.dilation, self.groups,
                                           self.deform_groups).to(memory_format=torch.channels_last)
        else:
            return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                           self.stride, self.padding,
                                           self.dilation, self.groups,
                                           self.deform_groups)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(
                    prefix[:-1] + '_offset.bias')
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
