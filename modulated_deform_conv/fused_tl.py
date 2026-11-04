"""
Fused Modulated Deformable Conv2d (forward) via tilelang implicit-GEMM — NHWC.

The im2col `columns` matrix is never materialized in global memory: the
deformable bilinear sampling + mask is done on-the-fly while staging the GEMM's
`block_M x block_K` tile into shared memory, then `T.gemm` accumulates into
tensorop fragments.

NHWC is chosen so the inner (channel) dim is contiguous: with K layout
`k = t*Cin + ci`, consecutive `k` threads read consecutive channels of the same
pixel -> coalesced bilinear loads, and the output `[B,Ho,Wo,Cout]` needs no
permute.

Scope (P1 of docs/fused_dcn_plan.md): groups==1, deform_groups==1, forward only.
Layout:
  data_im  [B, H, W, Cin]
  weight   [K, Cout]  with K = kH*kW*Cin,  k = (ki*kW+kj)*Cin + ci = t*Cin + ci
           (from torch weight [Cout,Cin,kH,kW]: .permute(2,3,1,0).reshape(K,Cout))
  offset   [B, Ho, Wo, 2*T]   (T = kH*kW, dg=1)
  mask     [B, Ho, Wo, T]
  output   [B, Ho, Wo, Cout]

GEMM dims:  M = B*Ho*Wo,  N = Cout,  K = kH*kW*Cin
"""

import torch
import tilelang
import tilelang.language as T
from tilelang import PassConfigKey

# The deformable gather builds a sampling coordinate `h_im = (int base) + offset`.
# tilelang's CSE hoists it into a Let whose integer-base bound the analyzer derives
# inconsistently (8 vs 10), tripping a bind-consistency assertion. Forcing Let
# inlining sidesteps the hoist. See investigation in git history.
_PASS_CFG = {PassConfigKey.TL_FORCE_LET_INLINE: True}


def out_size(H, pad, dil, k, stride):
    return (H + 2 * pad - (dil * (k - 1) + 1)) // stride + 1


@tilelang.jit(out_idx=[4], pass_configs=_PASS_CFG)
def _dcn_fwd_kernel_nhwc(
    B, Cin, H, W, Cout, kH, kW,
    sh, sw, ph, pw, dh, dw,
    block_M, block_N, block_K, num_stages, threads,
    dtype="float16", accum_dtype="float32",
):
    Tt = kH * kW
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    N = Cout
    K = Tt * Cin
    HoWo = Ho * Wo
    Kpad = T.ceildiv(K, block_K) * block_K
    Mpad = T.ceildiv(M, block_M) * block_M

    @T.prim_func
    def main(
        data_im: T.Tensor((B, H, W, Cin), dtype),
        weight:  T.Tensor((Kpad, N), dtype),            # [K, Cout] zero-padded on K
        offset:  T.Tensor((B, Ho, Wo, 2 * Tt), dtype),
        mask:    T.Tensor((B, Ho, Wo, Tt), dtype),
        out:     T.Tensor((Mpad, N), accum_dtype),       # [B*Ho*Wo, Cout] padded on M
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            col_s = T.alloc_shared((block_M, block_K), dtype)
            w_s   = T.alloc_shared((block_K, block_N), dtype)
            o_l   = T.alloc_fragment((block_M, block_N), accum_dtype)
            o_s   = T.alloc_shared((block_M, block_N), accum_dtype)

            T.clear(o_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = ko * block_K + j
                    in_range = (m < M) and (k < K)
                    ms = T.min(m, M - 1)
                    ks = T.min(k, K - 1)
                    # decode m -> (b, oh, ow)
                    b  = ms // HoWo
                    r  = ms % HoWo
                    oh = r // Wo
                    ow = r % Wo
                    # decode k -> (t, ci); t = ki*kW + kj
                    t  = ks // Cin
                    ci = ks % Cin
                    ki = t // kW
                    kj = t % kW

                    off_h = T.Cast(accum_dtype, offset[b, oh, ow, 2 * t])
                    off_w = T.Cast(accum_dtype, offset[b, oh, ow, 2 * t + 1])
                    msk   = T.Cast(accum_dtype, mask[b, oh, ow, t])

                    h_im = T.Cast(accum_dtype, oh * sh - ph + ki * dh) + off_h
                    w_im = T.Cast(accum_dtype, ow * sw - pw + kj * dw) + off_w

                    h_low = T.Cast("int32", T.floor(h_im))
                    w_low = T.Cast("int32", T.floor(w_im))
                    lh = h_im - T.Cast(accum_dtype, h_low)
                    lw = w_im - T.Cast(accum_dtype, w_low)
                    hh = T.Cast(accum_dtype, 1) - lh
                    hw = T.Cast(accum_dtype, 1) - lw

                    y0 = h_low
                    y1 = h_low + 1
                    x0 = w_low
                    x1 = w_low + 1
                    y0c = T.max(0, T.min(H - 1, y0))
                    y1c = T.max(0, T.min(H - 1, y1))
                    x0c = T.max(0, T.min(W - 1, x0))
                    x1c = T.max(0, T.min(W - 1, x1))
                    # Per-corner validity as float 0/1 — uniform across channels (ci),
                    # so the corner loads stay UNCONDITIONAL + contiguous in ci and the
                    # compiler can vectorize/coalesce them (a predicated per-element load
                    # is ~4x slower here). Indices are clamped so the loads are in-bounds.
                    g1 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                    g2 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                    g3 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                    g4 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                    ir = T.Cast(accum_dtype, in_range)
                    v1 = T.Cast(accum_dtype, data_im[b, y0c, x0c, ci])
                    v2 = T.Cast(accum_dtype, data_im[b, y0c, x1c, ci])
                    v3 = T.Cast(accum_dtype, data_im[b, y1c, x0c, ci])
                    v4 = T.Cast(accum_dtype, data_im[b, y1c, x1c, ci])
                    # NOTE: the golden's outer gate (h_im in (-1,H), w_im in (-1,W)) is
                    # mathematically redundant given per-corner zeroing.
                    val = (hh * hw * g1 * v1 + hh * lw * g2 * v2
                           + lh * hw * g3 * v3 + lh * lw * g4 * v4) * msk * ir
                    col_s[i, j] = T.Cast(dtype, val)

                T.copy(weight[ko * block_K, bx * block_N], w_s)
                T.gemm(col_s, w_s, o_l)

            T.copy(o_l, o_s)
            T.copy(o_s, out[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# NHWC pure-PyTorch golden (native, mirrors the kernel's K layout & bilinear)
# ---------------------------------------------------------------------------

def _bilinear_nhwc(im_b, h, w):
    """im_b [H,W,Cin]; h,w [P] float coords. Returns [P,Cin].

    Matches the CUDA/golden boundary behavior: each of the 4 corners is zeroed
    if its (iy,ix) falls outside [0,H-1]x[0,W-1]; outer gate h in (-1,H), w in (-1,W).
    """
    H, W, Cin = im_b.shape
    h_low = torch.floor(h).long()
    w_low = torch.floor(w).long()
    lh = h - h_low.to(h.dtype)
    lw = w - w_low.to(w.dtype)
    hh = 1.0 - lh
    hw = 1.0 - lw

    def corner(iy, ix):
        valid = (iy >= 0) & (iy <= H - 1) & (ix >= 0) & (ix <= W - 1)
        iy_c = iy.clamp(0, H - 1)
        ix_c = ix.clamp(0, W - 1)
        val = im_b[iy_c, ix_c]                       # [P, Cin]
        return torch.where(valid.unsqueeze(-1), val, torch.zeros_like(val))

    v1 = corner(h_low,     w_low)
    v2 = corner(h_low,     w_low + 1)
    v3 = corner(h_low + 1, w_low)
    v4 = corner(h_low + 1, w_low + 1)
    out = (hh * hw).unsqueeze(-1) * v1 + (hh * lw).unsqueeze(-1) * v2 \
        + (lh * hw).unsqueeze(-1) * v3 + (lh * lw).unsqueeze(-1) * v4
    outer = (h > -1) & (w > -1) & (h < H) & (w < W)
    return torch.where(outer.unsqueeze(-1), out, torch.zeros_like(out))


def dcn_columns_golden_nhwc(data_im, offset, mask, kH, kW, sh, sw, ph, pw, dh, dw):
    """Build the implicit-GEMM columns [M, K] in NHWC, K layout k = t*Cin + ci.

    data_im [B,H,W,Cin], offset [B,Ho,Wo,2T], mask [B,Ho,Wo,T].
    Returns columns [B*Ho*Wo, T*Cin].
    """
    B, H, W, Cin = data_im.shape
    Tt = kH * kW
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    dev, dt = data_im.device, data_im.dtype
    oh = torch.arange(Ho, device=dev, dtype=dt).view(Ho, 1).expand(Ho, Wo).reshape(-1)
    ow = torch.arange(Wo, device=dev, dtype=dt).view(1, Wo).expand(Ho, Wo).reshape(-1)
    cols = data_im.new_zeros(B, Ho * Wo, Tt, Cin)
    for b in range(B):
        for t in range(Tt):
            ki, kj = t // kW, t % kW
            off_h = offset[b, :, :, 2 * t].reshape(-1)
            off_w = offset[b, :, :, 2 * t + 1].reshape(-1)
            h_im = oh * sh - ph + ki * dh + off_h
            w_im = ow * sw - pw + kj * dw + off_w
            samp = _bilinear_nhwc(data_im[b], h_im, w_im)          # [Ho*Wo, Cin]
            msk = mask[b, :, :, t].reshape(-1, 1)
            cols[b, :, t, :] = samp * msk
    return cols.reshape(B * Ho * Wo, Tt * Cin)


def dcn_forward_golden_nhwc(data_im, offset, mask, weight, bias=None,
                            stride=1, padding=0, dilation=1):
    """Reference NHWC forward. Returns out [B,Ho,Wo,Cout]."""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    B, H, W, Cin = data_im.shape
    Cout, _, kH, kW = weight.shape
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    cols = dcn_columns_golden_nhwc(data_im, offset, mask, kH, kW, sh, sw, ph, pw, dh, dw)
    K = kH * kW * Cin
    w_flat = weight.permute(2, 3, 1, 0).reshape(K, Cout)           # [K, Cout]
    out = cols @ w_flat                                            # [M, Cout]
    out = out.view(B, Ho, Wo, Cout)
    if bias is not None:
        out = out + bias.view(1, 1, 1, -1)
    return out


# ---------------------------------------------------------------------------
# Kernel 1: offset conv (regular conv implicit-GEMM, NHWC) -> Y[M, 3T]
# ---------------------------------------------------------------------------

# Offset conv = REGULAR conv -> uses MUSA TME `T.c2d_im2col` (async bulk im2col via the
# Tensor Memory Engine, warp-specialized). ~3-4.6x faster than the manual gather for these
# shapes. Square kernel / uniform stride-dil-pad (dg=1 DCN). K-layout t*Cin+ci.
def _build_offset_conv_tme(B, Cin, H, W, Noff, kH, S, D, P, block_M, block_N, block_K,
                           num_stages, threads, dtype="float16", accum_dtype="float32"):
    Ho = (H + 2 * P - D * (kH - 1) - 1) // S + 1
    Wo = (W + 2 * P - D * (kH - 1) - 1) // S + 1
    M = B * Ho * Wo
    K = kH * kH * Cin
    Npad = ((Noff + block_N - 1) // block_N) * block_N
    Mpad = ((M + block_M - 1) // block_M) * block_M

    @T.prim_func
    def main(
        data:   T.Tensor((B, H, W, Cin), dtype),
        kernel: T.Tensor((kH, kH, Cin, Npad), dtype),    # [kH,kW,Cin,Noff(pad)], k=t*Cin+ci
        out:    T.Tensor((Mpad, Npad), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(Npad, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            x_s = T.alloc_shared((block_M, block_K), dtype)
            w_s = T.alloc_shared((block_K, block_N), dtype)
            o_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            kflat = T.Tensor((K, Npad), dtype, kernel.data)
            T.clear(o_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.c2d_im2col(data, x_s, by, ko, kH, S, D, P)
                T.copy(kflat[ko * block_K, bx * block_N], w_s)
                T.gemm(x_s, w_s, o_l)
            T.copy(o_l, out[by * block_M, bx * block_N])

    return tilelang.compile(main, out_idx=-1, target="musa", execution_backend="cython",
                            pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False})


# ---------------------------------------------------------------------------
# Deformable gather -> columns[M, K] in gmem (NHWC, K = t*Cin+ci).
# Used by the SPLIT path: tilelang gather + muBLAS torch.mm. On MUSA the tilelang
# T.gemm is ~2-2.6x slower than muBLAS for these shapes, so the fused implicit-GEMM
# loses to "gather to gmem + muBLAS mm" despite the extra column traffic. This split
# (all NHWC, permute-free) beats the existing CUDA path on all spec shapes.
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[3], pass_configs=_PASS_CFG)
def _dcn_gather_cols_kernel_nhwc(
    B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw,
    block_M, block_K, threads, dtype="float16", accum_dtype="float32",
):
    Tt = kH * kW
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    K = Tt * Cin
    HoWo = Ho * Wo
    Mpad = T.ceildiv(M, block_M) * block_M
    Kpad = T.ceildiv(K, block_K) * block_K

    @T.prim_func
    def main(
        data_im: T.Tensor((B, H, W, Cin), dtype),
        offset:  T.Tensor((B, Ho, Wo, 2 * Tt), dtype),
        mask:    T.Tensor((B, Ho, Wo, Tt), dtype),
        cols:    T.Tensor((Mpad, Kpad), dtype),       # [M, K] padded, k = t*Cin+ci
    ):
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            cs = T.alloc_shared((block_M, block_K), dtype)
            for i, j in T.Parallel(block_M, block_K):
                m = by * block_M + i
                k = bx * block_K + j
                in_range = (m < M) and (k < K)
                ms = T.min(m, M - 1)
                ks = T.min(k, K - 1)
                b  = ms // HoWo
                r  = ms % HoWo
                oh = r // Wo
                ow = r % Wo
                t  = ks // Cin
                ci = ks % Cin
                ki = t // kW
                kj = t % kW
                off_h = T.Cast(accum_dtype, offset[b, oh, ow, 2 * t])
                off_w = T.Cast(accum_dtype, offset[b, oh, ow, 2 * t + 1])
                msk   = T.Cast(accum_dtype, mask[b, oh, ow, t])
                h_im = T.Cast(accum_dtype, oh * sh - ph + ki * dh) + off_h
                w_im = T.Cast(accum_dtype, ow * sw - pw + kj * dw) + off_w
                h_low = T.Cast("int32", T.floor(h_im))
                w_low = T.Cast("int32", T.floor(w_im))
                lh = h_im - T.Cast(accum_dtype, h_low)
                lw = w_im - T.Cast(accum_dtype, w_low)
                hh = T.Cast(accum_dtype, 1) - lh
                hw = T.Cast(accum_dtype, 1) - lw
                y0 = h_low; y1 = h_low + 1; x0 = w_low; x1 = w_low + 1
                y0c = T.max(0, T.min(H - 1, y0)); y1c = T.max(0, T.min(H - 1, y1))
                x0c = T.max(0, T.min(W - 1, x0)); x1c = T.max(0, T.min(W - 1, x1))
                g1 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                g2 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                g3 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                g4 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                ir = T.Cast(accum_dtype, in_range)
                v1 = T.Cast(accum_dtype, data_im[b, y0c, x0c, ci])
                v2 = T.Cast(accum_dtype, data_im[b, y0c, x1c, ci])
                v3 = T.Cast(accum_dtype, data_im[b, y1c, x0c, ci])
                v4 = T.Cast(accum_dtype, data_im[b, y1c, x1c, ci])
                cs[i, j] = T.Cast(dtype, (hh * hw * g1 * v1 + hh * lw * g2 * v2
                                          + lh * hw * g3 * v3 + lh * lw * g4 * v4) * msk * ir)
            T.copy(cs, cols[by * block_M, bx * block_K])

    return main


_GCOLS_CACHE = {}


def dcn_gather_cols_nhwc(data_im, offset, mask, kH, kW, stride=1, padding=0, dilation=1,
                         block_M=64, block_K=64, threads=256):
    """Deformable gather -> columns [M, K] (NHWC, k = t*Cin+ci). M=B*Ho*Wo, K=kH*kW*Cin."""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    B, H, W, Cin = data_im.shape
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    K = kH * kW * Cin
    tl_dtype = _TL_DTYPE[data_im.dtype]
    key = (B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw, tl_dtype, block_M, block_K, threads)
    if key not in _GCOLS_CACHE:
        _GCOLS_CACHE[key] = _dcn_gather_cols_kernel_nhwc(
            B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw, block_M, block_K, threads,
            dtype=tl_dtype, accum_dtype="float32")
    cols = _GCOLS_CACHE[key](data_im.contiguous(), offset.contiguous(), mask.contiguous())
    return cols[:M, :K]


def dcn_forward_split_nhwc(data_im, w_off, b_off, weight, bias=None,
                           stride=1, padding=0, dilation=1):
    """RECOMMENDED full DCNv2 forward (NHWC, groups=1, dg=1): tilelang offset conv +
    tilelang deformable gather + muBLAS torch.mm. Beats the existing CUDA path on all
    spec shapes (A_B6 1.12x, A_B18 1.28x, B_B6 1.17x, B_B18 1.41x in fp16).

    data_im [B,H,W,Cin], w_off [3T,Cin,kH,kW], b_off [3T] or None,
    weight [Cout,Cin,kH,kW], bias [Cout] or None. Returns O [B,Ho,Wo,Cout].
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    B, H, W, Cin = data_im.shape
    Cout, _, kH, kW = weight.shape
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    K = kH * kW * Cin

    offset, mask = offset_conv_forward_nhwc(data_im, w_off, b_off, kH, kW, stride, padding, dilation)
    cols = dcn_gather_cols_nhwc(data_im, offset, mask, kH, kW, stride, padding, dilation)  # [M, K]
    w_flat = weight.permute(2, 3, 1, 0).reshape(K, Cout).contiguous().to(data_im.dtype)    # [K, Cout]
    out = torch.mm(cols, w_flat).view(B, Ho, Wo, Cout)
    if bias is not None:
        out = out + bias.view(1, 1, 1, -1)
    return out


_OFFSET_KERNEL_CACHE = {}


def _get_offset_kernel(B, Cin, H, W, Noff, kH, S, D, P, dtype,
                       block_M=64, block_N=32, block_K=32, num_stages=2, threads=256):
    key = (B, Cin, H, W, Noff, kH, S, D, P, dtype, block_M, block_N, block_K, num_stages, threads)
    if key not in _OFFSET_KERNEL_CACHE:
        _OFFSET_KERNEL_CACHE[key] = _build_offset_conv_tme(
            B, Cin, H, W, Noff, kH, S, D, P, block_M, block_N, block_K, num_stages, threads,
            dtype=dtype, accum_dtype="float32")
    return _OFFSET_KERNEL_CACHE[key]


def offset_conv_forward_nhwc(data_im, w_off, b_off, kH, kW,
                             stride=1, padding=0, dilation=1,
                             block_M=64, block_N=32, block_K=32, num_stages=2, threads=256):
    """Regular conv_offset (NHWC) via MUSA TME c2d_im2col -> (offset [B,Ho,Wo,2T], mask [B,Ho,Wo,T]).

    w_off [3T, Cin, kH, kW], b_off [3T] or None. Mirrors mmcv's pack:
      offset = Y[:, :2T] (= cat(o1,o2)),  mask = sigmoid(Y[:, 2T:3T]).
    Requires square kernel + uniform stride/dilation/pad (dg=1 DCN).
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    assert kH == kW and sh == sw and ph == pw and dh == dw, "TME c2d_im2col needs square/uniform conv"
    B, H, W, Cin = data_im.shape
    Noff, Cin_w, _, _ = w_off.shape
    Tt = kH * kW
    assert Noff == 3 * Tt and Cin_w == Cin
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo

    tl_dtype = _TL_DTYPE[data_im.dtype]
    Npad = ((Noff + block_N - 1) // block_N) * block_N
    # kernel as [kH,kW,Cin,Npad], k = t*Cin+ci (from w_off [Noff,Cin,kH,kW])
    kern = w_off.permute(2, 3, 1, 0).contiguous().to(data_im.dtype)        # [kH,kW,Cin,Noff]
    if Npad != Noff:
        kp = kern.new_zeros(kH, kW, Cin, Npad)
        kp[..., :Noff] = kern
        kern = kp

    kernel = _get_offset_kernel(B, Cin, H, W, Noff, kH, sh, dh, ph, tl_dtype,
                                block_M, block_N, block_K, num_stages, threads)
    y_pad = kernel(data_im.contiguous(), kern)            # [Mpad, Npad] fp32
    y = y_pad[:M, :Noff]                                  # [M, 3T]
    if b_off is not None:
        y = y + b_off.view(1, -1).to(y.dtype)
    y = y.view(B, Ho, Wo, Noff)
    offset = y[..., : 2 * Tt].contiguous().to(data_im.dtype)
    mask = torch.sigmoid(y[..., 2 * Tt:]).contiguous().to(data_im.dtype)
    return offset, mask


def fused_dcn_full_forward_nhwc(data_im, w_off, b_off, weight, bias=None,
                                stride=1, padding=0, dilation=1):
    """Full DCNv2 forward (NHWC): offset conv (kernel1) + deform sample/main GEMM (kernel2).

    data_im [B,H,W,Cin], w_off [3T,Cin,kH,kW], b_off [3T] or None,
    weight [Cout,Cin,kH,kW], bias [Cout] or None.  Returns O [B,Ho,Wo,Cout].
    """
    _, _, kH, kW = weight.shape
    offset, mask = offset_conv_forward_nhwc(data_im, w_off, b_off, kH, kW,
                                            stride, padding, dilation)
    return fused_dcn_forward_nhwc(data_im, offset, mask, weight, bias,
                                  stride, padding, dilation)


_KERNEL_CACHE = {}


def _get_kernel(B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, dtype,
                block_M=32, block_N=256, block_K=32, num_stages=2, threads=256):
    key = (B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, dtype,
           block_M, block_N, block_K, num_stages, threads)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _dcn_fwd_kernel_nhwc(
            B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw,
            block_M, block_N, block_K, num_stages, threads,
            dtype=dtype, accum_dtype="float32",
        )
    return _KERNEL_CACHE[key]


_TL_DTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def weight_to_nhwc_flat(weight, block_K=32):
    """torch weight [Cout,Cin,kH,kW] -> [Kpad, Cout], k = (ki*kW+kj)*Cin+ci."""
    Cout, Cin, kH, kW = weight.shape
    K = kH * kW * Cin
    w_flat = weight.permute(2, 3, 1, 0).reshape(K, Cout).contiguous()   # [K, Cout]
    Kpad = ((K + block_K - 1) // block_K) * block_K
    if Kpad != K:
        w_pad = w_flat.new_zeros(Kpad, Cout)
        w_pad[:K] = w_flat
        w_flat = w_pad
    return w_flat


def fused_dcn_forward_nhwc(data_im, offset, mask, weight, bias=None,
                           stride=1, padding=0, dilation=1,
                           block_M=32, block_N=256, block_K=32, num_stages=2, threads=256,
                           return_flat=False):
    """Forward modulated deformable conv2d (groups=1, dg=1), NHWC, via tilelang.

    data_im [B,H,W,Cin], offset [B,Ho,Wo,2T], mask [B,Ho,Wo,T],
    weight  [Cout,Cin,kH,kW], bias [Cout] or None.
    Returns out [B,Ho,Wo,Cout].
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    B, H, W, Cin = data_im.shape
    Cout, Cin_w, kH, kW = weight.shape
    assert Cin_w == Cin, "groups>1 not supported in fused_tl"
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo

    tl_dtype = _TL_DTYPE[data_im.dtype]
    w_flat = weight_to_nhwc_flat(weight, block_K).to(data_im.dtype)

    kernel = _get_kernel(B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw,
                         tl_dtype, block_M, block_N, block_K, num_stages, threads)

    out_pad = kernel(data_im.contiguous(), w_flat,
                     offset.contiguous(), mask.contiguous())   # [Mpad, Cout] fp32
    out = out_pad[:M].to(data_im.dtype)
    if return_flat:
        if bias is not None:
            out = out + bias.view(1, -1)
        return out                                            # [B*Ho*Wo, Cout]
    out = out.view(B, Ho, Wo, Cout)
    if bias is not None:
        out = out + bias.view(1, 1, 1, -1)
    return out


# ---------------------------------------------------------------------------
# Single fused operator: offset conv + sigmoid + deform sample + main GEMM,
# all in ONE kernel launch. Grid is over M only (one N-block = Cout) so the
# offset conv is computed once per pixel-tile and offset/mask stay in shared.
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[4], pass_configs=_PASS_CFG)
def _dcn_mega_kernel_nhwc(
    B, Cin, H, W, Cout, kH, kW,
    sh, sw, ph, pw, dh, dw,
    block_M, block_N, block_K, num_stages, threads,
    dtype="float16", accum_dtype="float32",
):
    Tt = kH * kW
    Noff = 3 * Tt
    Noff_pad = ((Noff + 15) // 16) * 16          # tensorop-friendly small N
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    K = Tt * Cin
    HoWo = Ho * Wo
    # block_N must satisfy block_N*elem_bytes <= 512 (tensorop N-tile cap on mp_31).
    # 2D grid over (N-block, M-block); offset conv is independent of the N-block (bx)
    # so it is recomputed per N-block when Cout > block_N (acceptable; offset conv is
    # the cheap part). The gather IS recomputed per N-block too -> prefer block_N=Cout
    # when it fits the cap (e.g. Cout=256 fp16).
    Kpad = T.ceildiv(K, block_K) * block_K
    Mpad = T.ceildiv(M, block_M) * block_M

    @T.prim_func
    def main(
        data_im: T.Tensor((B, H, W, Cin), dtype),
        w_off:   T.Tensor((Kpad, Noff_pad), dtype),    # offset-conv weight [K,3T] padded
        b_off:   T.Tensor((Noff_pad,), accum_dtype),   # offset-conv bias padded
        w_main:  T.Tensor((Kpad, Cout), dtype),        # main weight [K,Cout] padded on K
        out:     T.Tensor((Mpad, Cout), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(Cout, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            x_s     = T.alloc_shared((block_M, block_K), dtype)
            woff_s  = T.alloc_shared((block_K, Noff_pad), dtype)
            wmain_s = T.alloc_shared((block_K, block_N), dtype)
            col_s   = T.alloc_shared((block_M, block_K), dtype)
            y_f     = T.alloc_fragment((block_M, Noff_pad), accum_dtype)
            y_s     = T.alloc_shared((block_M, Noff_pad), accum_dtype)
            o_l     = T.alloc_fragment((block_M, block_N), accum_dtype)

            # ---- Phase 1: offset conv (regular implicit-GEMM) -> y_f[block_M, 3T] ----
            T.clear(y_f)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = ko * block_K + j
                    in_range = (m < M) and (k < K)
                    ms = T.min(m, M - 1)
                    ks = T.min(k, K - 1)
                    b  = ms // HoWo
                    r  = ms % HoWo
                    oh = r // Wo
                    ow = r % Wo
                    t  = ks // Cin
                    ci = ks % Cin
                    ki = t // kW
                    kj = t % kW
                    yy = oh * sh - ph + ki * dh
                    xx = ow * sw - pw + kj * dw
                    valid = in_range and (yy >= 0) and (yy < H) and (xx >= 0) and (xx < W)
                    x_s[i, j] = T.if_then_else(valid, data_im[b, yy, xx, ci], T.Cast(dtype, 0))
                T.copy(w_off[ko * block_K, 0], woff_s)
                T.gemm(x_s, woff_s, y_f)
            T.copy(y_f, y_s)
            # add offset-conv bias once (offset+mask raw both live in y_s)
            for i, c in T.Parallel(block_M, Noff_pad):
                y_s[i, c] = y_s[i, c] + b_off[c]

            # ---- Phase 2: deform sample (uses y_s for offset/mask) + main GEMM ----
            T.clear(o_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = ko * block_K + j
                    in_range = (m < M) and (k < K)
                    ms = T.min(m, M - 1)
                    ks = T.min(k, K - 1)
                    b  = ms // HoWo
                    r  = ms % HoWo
                    oh = r // Wo
                    ow = r % Wo
                    t  = ks // Cin
                    ci = ks % Cin
                    ki = t // kW
                    kj = t % kW

                    off_h = y_s[i, 2 * t]
                    off_w = y_s[i, 2 * t + 1]
                    msk   = T.sigmoid(y_s[i, 2 * Tt + t])

                    h_im = T.Cast(accum_dtype, oh * sh - ph + ki * dh) + off_h
                    w_im = T.Cast(accum_dtype, ow * sw - pw + kj * dw) + off_w
                    h_low = T.Cast("int32", T.floor(h_im))
                    w_low = T.Cast("int32", T.floor(w_im))
                    lh = h_im - T.Cast(accum_dtype, h_low)
                    lw = w_im - T.Cast(accum_dtype, w_low)
                    hh = T.Cast(accum_dtype, 1) - lh
                    hw = T.Cast(accum_dtype, 1) - lw
                    y0 = h_low
                    y1 = h_low + 1
                    x0 = w_low
                    x1 = w_low + 1
                    y0c = T.max(0, T.min(H - 1, y0))
                    y1c = T.max(0, T.min(H - 1, y1))
                    x0c = T.max(0, T.min(W - 1, x0))
                    x1c = T.max(0, T.min(W - 1, x1))
                    g1 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                    g2 = T.Cast(accum_dtype, (y0 >= 0) and (y0 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                    g3 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x0 >= 0) and (x0 <= W - 1))
                    g4 = T.Cast(accum_dtype, (y1 >= 0) and (y1 <= H - 1) and (x1 >= 0) and (x1 <= W - 1))
                    ir = T.Cast(accum_dtype, in_range)
                    v1 = T.Cast(accum_dtype, data_im[b, y0c, x0c, ci])
                    v2 = T.Cast(accum_dtype, data_im[b, y0c, x1c, ci])
                    v3 = T.Cast(accum_dtype, data_im[b, y1c, x0c, ci])
                    v4 = T.Cast(accum_dtype, data_im[b, y1c, x1c, ci])
                    val = (hh * hw * g1 * v1 + hh * lw * g2 * v2
                           + lh * hw * g3 * v3 + lh * lw * g4 * v4) * msk * ir
                    col_s[i, j] = T.Cast(dtype, val)
                T.copy(w_main[ko * block_K, bx * block_N], wmain_s)
                T.gemm(col_s, wmain_s, o_l)

            T.copy(o_l, out[by * block_M, bx * block_N])

    return main


_MEGA_CACHE = {}


def _get_mega_kernel(B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, dtype,
                     block_M=32, block_N=128, block_K=64, num_stages=2, threads=256):
    key = (B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw, dtype,
           block_M, block_N, block_K, num_stages, threads)
    if key not in _MEGA_CACHE:
        _MEGA_CACHE[key] = _dcn_mega_kernel_nhwc(
            B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw,
            block_M, block_N, block_K, num_stages, threads,
            dtype=dtype, accum_dtype="float32",
        )
    return _MEGA_CACHE[key]


def fused_dcn_mega_forward_nhwc(data_im, w_off, b_off, weight, bias=None,
                                stride=1, padding=0, dilation=1,
                                block_M=32, block_N=None, block_K=64, num_stages=2, threads=256):
    """Single-kernel full DCNv2 forward (NHWC, groups=1, dg=1).

    data_im [B,H,W,Cin], w_off [3T,Cin,kH,kW], b_off [3T] or None,
    weight [Cout,Cin,kH,kW], bias [Cout] or None. Returns O [B,Ho,Wo,Cout].

    block_N defaults to min(Cout, cap) where cap=256 (fp16/bf16) or 128 (fp32),
    respecting the tensorop N-tile cap (block_N*elem_bytes <= 512).
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    B, H, W, Cin = data_im.shape
    Cout, Cin_w, kH, kW = weight.shape
    assert Cin_w == Cin, "groups>1 not supported"
    Tt = kH * kW
    Noff = 3 * Tt
    Noff_pad = ((Noff + 15) // 16) * 16
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    K = kH * kW * Cin
    Kpad = ((K + block_K - 1) // block_K) * block_K
    tl_dtype = _TL_DTYPE[data_im.dtype]

    if block_N is None:
        cap = 512 // (torch.finfo(data_im.dtype).bits // 8)   # tensorop N-tile cap
        block_N = Cout
        while block_N > cap or Cout % block_N != 0:
            block_N //= 2

    # offset weight [K, Noff] -> pad to [Kpad, Noff_pad]
    wo = w_off.permute(2, 3, 1, 0).reshape(K, Noff).contiguous().to(data_im.dtype)
    wo_p = wo.new_zeros(Kpad, Noff_pad)
    wo_p[:K, :Noff] = wo
    bo_p = torch.zeros(Noff_pad, device=data_im.device, dtype=torch.float32)
    if b_off is not None:
        bo_p[:Noff] = b_off.float()
    # main weight [K, Cout] -> pad to [Kpad, Cout]
    wm = weight.permute(2, 3, 1, 0).reshape(K, Cout).contiguous().to(data_im.dtype)
    if Kpad != K:
        wm_p = wm.new_zeros(Kpad, Cout)
        wm_p[:K] = wm
        wm = wm_p

    kernel = _get_mega_kernel(B, Cin, H, W, Cout, kH, kW, sh, sw, ph, pw, dh, dw,
                              tl_dtype, block_M, block_N, block_K, num_stages, threads)
    out_pad = kernel(data_im.contiguous(), wo_p, bo_p, wm)     # [Mpad, Cout] fp32
    out = out_pad[:M].to(data_im.dtype).view(B, Ho, Wo, Cout)
    if bias is not None:
        out = out + bias.view(1, 1, 1, -1)
    return out


# ---------------------------------------------------------------------------
# Backward: deformable col2im / col2im_coord (NHWC). Given dCol[M,K], X, offset,
# mask -> grad_offset[B,Ho,Wo,2T], grad_mask[B,Ho,Wo,T] (reduce over ci) and
# grad_X_samp[B,H,W,Cin] (4-corner scatter via atomic_add). grad_X MUST be a
# pre-zeroed input (atomic accumulator), not an out_idx tensor.
# ---------------------------------------------------------------------------

# Backward col2im (NHWC): ONE fused (m,t,ci)-parallel pass. grad_X scattered to its 4
# corners via atomic_add; grad_offset/grad_mask reduced over ci via warp_reduce_sum
# (a warp = 32 contiguous ci of one (m,t)) then ONE atomic per warp. gX/gOff/gMask are all
# PRE-ZEROED atomic accumulators. MUST NOT use TL_FORCE_LET_INLINE: inlining the reduce
# operand makes warp_reduce_sum recompute it across shuffle steps -> wrong results.
_COL2IM_PASS_CFG = {PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}


@tilelang.jit(pass_configs=_COL2IM_PASS_CFG)
def _dcn_col2im_kernel_nhwc(B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw, block,
                            dtype="float32", accum_dtype="float32"):
    Tt = kH * kW
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    M = B * Ho * Wo
    K = Tt * Cin
    HoWo = Ho * Wo
    MTC = M * K

    @T.prim_func
    def main(
        dCol:   T.Tensor((M, K), dtype),
        X:      T.Tensor((B, H, W, Cin), dtype),
        offset: T.Tensor((B, Ho, Wo, 2 * Tt), dtype),
        mask:   T.Tensor((B, Ho, Wo, Tt), dtype),
        gX:     T.Tensor((B, H, W, Cin), accum_dtype),
        gOff:   T.Tensor((B, Ho, Wo, 2 * Tt), accum_dtype),
        gMask:  T.Tensor((B, Ho, Wo, Tt), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(MTC, block), threads=block) as blk:
            tx = T.get_thread_binding(0)
            idx = blk * block + tx
            ir = idx < MTC
            s = T.min(idx, MTC - 1)
            m = s // K
            k = s % K
            t = k // Cin
            ci = k % Cin
            ki = t // kW
            kj = t % kW
            b = m // HoWo
            r = m % HoWo
            oh = r // Wo
            ow = r % Wo
            off_h = offset[b, oh, ow, 2 * t]
            off_w = offset[b, oh, ow, 2 * t + 1]
            msk = mask[b, oh, ow, t]
            h_im = T.Cast(accum_dtype, oh * sh - ph + ki * dh) + off_h
            w_im = T.Cast(accum_dtype, ow * sw - pw + kj * dw) + off_w
            hl = T.Cast("int32", T.floor(h_im))
            wl = T.Cast("int32", T.floor(w_im))
            lh = h_im - T.Cast(accum_dtype, hl)
            lw = w_im - T.Cast(accum_dtype, wl)
            hh = T.Cast(accum_dtype, 1) - lh
            hw = T.Cast(accum_dtype, 1) - lw
            y0 = hl; y1 = hl + 1; x0 = wl; x1 = wl + 1
            y0c = T.max(0, T.min(H - 1, y0)); y1c = T.max(0, T.min(H - 1, y1))
            x0c = T.max(0, T.min(W - 1, x0)); x1c = T.max(0, T.min(W - 1, x1))
            g1 = T.Cast(accum_dtype, (y0 >= 0) and (y0 < H) and (x0 >= 0) and (x0 < W))
            g2 = T.Cast(accum_dtype, (y0 >= 0) and (y0 < H) and (x1 >= 0) and (x1 < W))
            g3 = T.Cast(accum_dtype, (y1 >= 0) and (y1 < H) and (x0 >= 0) and (x0 < W))
            g4 = T.Cast(accum_dtype, (y1 >= 0) and (y1 < H) and (x1 >= 0) and (x1 < W))
            v1 = X[b, y0c, x0c, ci]; v2 = X[b, y0c, x1c, ci]
            v3 = X[b, y1c, x0c, ci]; v4 = X[b, y1c, x1c, ci]
            dc = dCol[m, k]
            irf = T.Cast(accum_dtype, ir)
            S = hh * hw * g1 * v1 + hh * lw * g2 * v2 + lh * hw * g3 * v3 + lh * lw * g4 * v4
            dSf = dc * msk
            # grad_X scatter (per-thread)
            T.atomic_add(gX[b, y0c, x0c, ci], dSf * hh * hw * g1 * irf)
            T.atomic_add(gX[b, y0c, x1c, ci], dSf * hh * lw * g2 * irf)
            T.atomic_add(gX[b, y1c, x0c, ci], dSf * lh * hw * g3 * irf)
            T.atomic_add(gX[b, y1c, x1c, ci], dSf * lh * lw * g4 * irf)
            # grad_offset/grad_mask: warp-reduce over the warp's 32 ci, one atomic per warp
            sh_ = T.warp_reduce_sum(dSf * (-hw * g1 * v1 - lw * g2 * v2 + hw * g3 * v3 + lw * g4 * v4) * irf)
            sw_ = T.warp_reduce_sum(dSf * (-hh * g1 * v1 + hh * g2 * v2 - lh * g3 * v3 + lh * g4 * v4) * irf)
            sm_ = T.warp_reduce_sum(dc * S * irf)
            if tx % 32 == 0:
                T.atomic_add(gOff[b, oh, ow, 2 * t], sh_)
                T.atomic_add(gOff[b, oh, ow, 2 * t + 1], sw_)
                T.atomic_add(gMask[b, oh, ow, t], sm_)

    return main


_COL2IM_CACHE = {}


def dcn_col2im_nhwc(dCol, X, offset, mask, kH, kW, stride=1, padding=0, dilation=1, block=256):
    """Fused backward col2im + col2im_coord. Returns (grad_X, grad_offset, grad_mask), all fp32."""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    B, H, W, Cin = X.shape
    Ho = out_size(H, ph, dh, kH, sh)
    Wo = out_size(W, pw, dw, kW, sw)
    Tt = kH * kW
    key = (B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw, block)
    if key not in _COL2IM_CACHE:
        _COL2IM_CACHE[key] = _dcn_col2im_kernel_nhwc(
            B, Cin, H, W, kH, kW, sh, sw, ph, pw, dh, dw, block,
            dtype="float32", accum_dtype="float32")
    dCf = dCol.float().contiguous(); Xf = X.float().contiguous()
    offf = offset.float().contiguous(); mkf = mask.float().contiguous()
    gX = torch.zeros(B, H, W, Cin, device=X.device, dtype=torch.float32)
    gOff = torch.zeros(B, Ho, Wo, 2 * Tt, device=X.device, dtype=torch.float32)
    gMask = torch.zeros(B, Ho, Wo, Tt, device=X.device, dtype=torch.float32)
    _COL2IM_CACHE[key](dCf, Xf, offf, mkf, gX, gOff, gMask)
    return gX, gOff, gMask




class _DCNSplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_nhwc, w_off, b_off, weight, bias, stride, padding, dilation):
        out = dcn_forward_split_nhwc(x_nhwc, w_off, b_off, weight, bias, stride, padding, dilation)
        # recompute offset/mask for backward (cheap; avoids saving big tensors)
        _, _, kH, kW = weight.shape
        offset, mask = offset_conv_forward_nhwc(x_nhwc, w_off, b_off, kH, kW, stride, padding, dilation)
        ctx.save_for_backward(x_nhwc, offset, mask, w_off, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        ctx.with_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        import torch.nn.grad as tnng
        x, offset, mask, w_off, weight = ctx.saved_tensors
        st, pd, dl = ctx.stride, ctx.padding, ctx.dilation
        stt = (st, st) if isinstance(st, int) else st
        pdt = (pd, pd) if isinstance(pd, int) else pd
        dlt = (dl, dl) if isinstance(dl, int) else dl
        B, H, W, Cin = x.shape
        Cout, _, kH, kW = weight.shape
        Tt = kH * kW
        Ho = out_size(H, pdt[0], dlt[0], kH, stt[0])
        Wo = out_size(W, pdt[1], dlt[1], kW, stt[1])
        M = B * Ho * Wo
        K = kH * kW * Cin
        dt = x.dtype

        dO = grad_out.reshape(M, Cout).to(dt)
        w_flat = weight.permute(2, 3, 1, 0).reshape(K, Cout).contiguous().to(dt)   # [K, Cout]
        dCol = torch.mm(dO, w_flat.t())                                            # [M, K]

        gX_samp, gOff, gMask = dcn_col2im_nhwc(dCol, x, offset, mask, kH, kW, st, pd, dl)

        # dW_main = Col^T @ dO  (recompute Col)
        cols = dcn_gather_cols_nhwc(x, offset, mask, kH, kW, st, pd, dl)            # [M, K]
        dW_flat = torch.mm(cols.t().float(), dO.float())                           # [K, Cout]
        grad_weight = dW_flat.view(kH, kW, Cin, Cout).permute(3, 2, 0, 1).contiguous().to(dt)
        grad_bias = dO.float().sum(0).to(dt) if ctx.with_bias else None

        # sigmoid backward: mask = sigmoid(mask_raw) -> grad_mask_raw = gMask * mask*(1-mask)
        m_f = mask.float()
        gmask_raw = gMask * m_f * (1.0 - m_f)                                      # [B,Ho,Wo,T]

        # conv_offset backward (regular conv): dY = concat(gOff, gmask_raw) NHWC -> NCHW
        dY = torch.cat([gOff, gmask_raw], dim=-1).permute(0, 3, 1, 2).contiguous() # [B,3T,Ho,Wo]
        x_nchw = x.permute(0, 3, 1, 2).contiguous().float()
        w_off_f = w_off.float()
        grad_x_off = tnng.conv2d_input(x_nchw.shape, w_off_f, dY, stride=stt, padding=pdt, dilation=dlt)
        grad_w_off = tnng.conv2d_weight(x_nchw, w_off_f.shape, dY, stride=stt, padding=pdt, dilation=dlt)
        grad_b_off = dY.sum(dim=(0, 2, 3))
        grad_x_off = grad_x_off.permute(0, 2, 3, 1)                                # NHWC

        grad_x = (gX_samp + grad_x_off).to(dt)
        return (grad_x, grad_w_off.to(dt), grad_b_off.to(dt), grad_weight, grad_bias,
                None, None, None)


def dcn_split(x_nhwc, w_off, b_off, weight, bias=None, stride=1, padding=0, dilation=1):
    """Autograd-enabled full DCNv2 forward (NHWC, split path). Differentiable."""
    return _DCNSplitFunction.apply(x_nhwc, w_off, b_off, weight, bias, stride, padding, dilation)


def dcn_apply_nchw(x_nchw, w_off, b_off, weight, bias=None, stride=1, padding=0, dilation=1):
    """NCHW-in/NCHW-out adapter over the NHWC tilelang DCNv2 (autograd-enabled).

    Lets the tilelang backend be a drop-in for the NCHW CUDA path: input/output and
    weights stay in the standard NCHW / [Cout,Cin,kH,kW] layouts; the NHWC permutes
    are differentiable so gradients flow back to all params.
    Scope: groups==1, deform_groups==1, square kernel, uniform stride/pad/dilation.
    """
    # channels_last NCHW -> permute(0,2,3,1) is a free contiguous NHWC view (no copy);
    # plain-contiguous NCHW needs a real transpose. Match output format to input so a
    # channels_last pipeline pays zero permute cost on either side.
    is_cl = x_nchw.is_contiguous(memory_format=torch.channels_last)
    x_nhwc = x_nchw.permute(0, 2, 3, 1)
    if not is_cl:
        x_nhwc = x_nhwc.contiguous()
    out_nhwc = dcn_split(x_nhwc, w_off, b_off, weight, bias, stride, padding, dilation)  # [B,Ho,Wo,Cout]
    out = out_nhwc.permute(0, 3, 1, 2)                 # [B,Cout,Ho,Wo] (channels_last view)
    return out if is_cl else out.contiguous()
