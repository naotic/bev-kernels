/*
 * Fused Modulated Deformable Conv2d — NHWC hand-written CUDA kernels.
 *
 * Layout (groups==1, deformable_group==1):
 *   data_im   [B, H, W, Cin]
 *   offset    [B, Ho, Wo, 2T]   (T = kH*kW)
 *   mask      [B, Ho, Wo, T]
 *   data_col  [M, T*Cin]        M = B*Ho*Wo,  k = t*Cin + ci  (ci fastest)
 *
 * The main GEMM (Col @ Wflat) and the offset conv are kept external (torch /
 * muBLAS / muDNN). This file only owns the deformable gather (and, later, the
 * NHWC backward). NHWC makes the inner channel dim contiguous so the 32 lanes
 * of a warp read/write consecutive channels -> coalesced.
 *
 * Gather parallelization (the lever vs a naive per-element scheme):
 *   one warp owns one (m, t) pair; lanes iterate channels ci = lane, lane+32...
 *   All the geometry (m/t decode, offset/mask read, floor, 4 corners, boundary
 *   validity, bilinear weights) is computed ONCE per (m,t) as uniform warp work
 *   and reused across all Cin channels — so the per-(m,t) scalar arithmetic is
 *   amortized by a factor of ~Cin/32 instead of being redone for every channel.
 *   The per-channel inner loop is then just 4 coalesced loads + 4 FMA + 1 store.
 */

#ifndef DCN_FUSED_NHWC_KERNEL_CUH
#define DCN_FUSED_NHWC_KERNEL_CUH

#include <float.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace dcn_nhwc {

constexpr int WARP_SIZE = 32;

// Aligned 2-wide POD so a pair of contiguous channels loads/stores as one wide
// (32-bit for half/bf16, 64-bit for float) transaction. Portable — no reliance
// on dtype-specific vector intrinsics (__half2 etc.) that mcc/torchada may rename.
template <typename T>
struct alignas(2 * sizeof(T)) Vec2 {
  T x, y;
};

// Generic aligned N-wide POD (N in {2,4}) -> one wide coalesced load/store covering
// N contiguous channels per lane. alignas gives the compiler the aligned vector op.
template <typename T, int N>
struct alignas(N * sizeof(T)) VecN {
  T v[N];
};

// Butterfly all-reduce: every lane ends with the warp-wide sum.
__device__ __forceinline__ float warp_all_sum(float v) {
#pragma unroll
  for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.f / (1.f + __expf(-x));
}

// Deformable gather, NHWC, dg==1. data_col[m, t*Cin+ci] = mask * bilinear(X, loc).
//
// Coordinates (b, oh, ow, t) come straight from the grid — NOT decoded from a
// flat index — so there are no runtime integer divisions by Ho/Wo/HoWo (the
// expensive part: GPU int-div is emulated, and these divisors are runtime so the
// compiler can't strength-reduce them). Grid layout:
//   blockIdx.z = b * Ttaps + t        (b,t; KH*KW is a compile-time literal -> reduced)
//   blockIdx.y = oh
//   blockIdx.x = ow tile;  warp_in_block selects ow within the tile
//   lane (0..31) iterates channels ci = lane, lane+32, ...
// KH/KW are template params (BEVFormer is 3x3); KH==0 selects a runtime fallback.
//
// Boundary semantics match the pure-PyTorch golden: each of the 4 corners is
// zeroed if it falls outside [0,H-1]x[0,W-1] (per-corner gating). The golden's
// outer gate (h in (-1,H), w in (-1,W)) is mathematically redundant given the
// per-corner zeroing, so it is omitted. Indices are clamped so the (zero-weighted)
// loads stay in-bounds, keeping them unconditional + coalesced across channels.
// 32-bit indexing throughout (Col = M*T*Cin stays < 2^31, checked in launcher).
// VW = channels processed per lane per step (vector width, 2 or 4).
// FUSED_Y=true: data_offset is the raw offset-conv output y [B,Ho,Wo,3T]; offset =
// y[...,0:2T], mask = sigmoid(y[...,2T:3T]) read on the fly (fuses the fwd sigmoid
// + avoids materializing/contiguous-copying separate offset/mask). data_mask unused.
template <typename T, int KH, int KW, int VW, bool FUSED_Y>
__global__ void modulated_deformable_im2col_nhwc_gpu_kernel(
    const T *__restrict__ data_im,      // [B, H, W, Cin]
    const T *__restrict__ data_offset,  // [B,Ho,Wo,2T]  or  y [B,Ho,Wo,3T] if FUSED_Y
    const T *__restrict__ data_mask,    // [B, Ho, Wo, T] (unused if FUSED_Y)
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int kH_rt, const int kW_rt,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dil_h, const int dil_w,
    T *__restrict__ data_col) {         // [M, T*Cin], M = B*Ho*Wo
  constexpr bool DYN = (KH == 0);
  const int kw = DYN ? kW_rt : KW;
  const int Ttaps = DYN ? (kH_rt * kW_rt) : (KH * KW);

  const int WPB = blockDim.x >> 5;           // warps per block
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int ow = blockIdx.x * WPB + (threadIdx.x >> 5);
  if (ow >= Wo) return;
  const int oh = blockIdx.y;

  int b, t;
  if constexpr (DYN) {
    b = blockIdx.z / Ttaps;
    t = blockIdx.z - b * Ttaps;
  } else {
    b = blockIdx.z / (KH * KW);           // KH*KW literal -> strength-reduced
    t = blockIdx.z - b * (KH * KW);
  }
  int ki, kj;
  if constexpr (DYN) {
    ki = t / kw;
    kj = t - ki * kw;
  } else {
    ki = t / KW;                          // KW literal -> strength-reduced
    kj = t - ki * KW;
  }

  const int m = (b * Ho + oh) * Wo + ow;
  // offset/mask are [B,Ho,Wo,*] — channel-last in the tap dim
  const int pix = m;
  float off_h, off_w, msk;
  if constexpr (FUSED_Y) {  // data_offset == y [B,Ho,Wo,3T]
    const int yb = pix * (3 * Ttaps);
    off_h = (float)data_offset[yb + 2 * t];
    off_w = (float)data_offset[yb + 2 * t + 1];
    msk = sigmoidf((float)data_offset[yb + 2 * Ttaps + t]);
  } else {
    off_h = (float)data_offset[pix * (2 * Ttaps) + 2 * t];
    off_w = (float)data_offset[pix * (2 * Ttaps) + 2 * t + 1];
    msk = (float)data_mask[pix * Ttaps + t];
  }

  const float h_im = (float)(oh * stride_h - pad_h + ki * dil_h) + off_h;
  const float w_im = (float)(ow * stride_w - pad_w + kj * dil_w) + off_w;

  const int h_low = (int)floorf(h_im);
  const int w_low = (int)floorf(w_im);
  const float lh = h_im - (float)h_low;
  const float lw = w_im - (float)w_low;
  const float hh = 1.f - lh, hw = 1.f - lw;

  const int y0 = h_low, y1 = h_low + 1;
  const int x0 = w_low, x1 = w_low + 1;
  const bool g1 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W);
  const bool g2 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W);
  const bool g3 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W);
  const bool g4 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W);
  const int y0c = min(max(y0, 0), H - 1), y1c = min(max(y1, 0), H - 1);
  const int x0c = min(max(x0, 0), W - 1), x1c = min(max(x1, 0), W - 1);

  // pre-combined uniform corner weights (fold mask + per-corner gate in)
  const float cw1 = hh * hw * (g1 ? msk : 0.f);
  const float cw2 = hh * lw * (g2 ? msk : 0.f);
  const float cw3 = lh * hw * (g3 ? msk : 0.f);
  const float cw4 = lh * lw * (g4 ? msk : 0.f);

  // channel-contiguous base offsets for the 4 corners: X[b, y, x, ci]
  const int base_b = b * H * W * Cin;
  const int p1 = base_b + (y0c * W + x0c) * Cin;
  const int p2 = base_b + (y0c * W + x1c) * Cin;
  const int p3 = base_b + (y1c * W + x0c) * Cin;
  const int p4 = base_b + (y1c * W + x1c) * Cin;

  const int col_base = m * (Ttaps * Cin) + t * Cin;  // k = t*Cin+ci

  // Vectorized inner loop: VW channels per lane via wide aligned load/store.
  // p1..p4 and col_base are multiples of Cin (mult. of VW for all target shapes)
  // so the VecN reinterpret stays aligned. A scalar tail covers the remainder.
  using V = VecN<T, VW>;
  const int CinV = Cin / VW;
  const V *im1 = reinterpret_cast<const V *>(data_im + p1);
  const V *im2 = reinterpret_cast<const V *>(data_im + p2);
  const V *im3 = reinterpret_cast<const V *>(data_im + p3);
  const V *im4 = reinterpret_cast<const V *>(data_im + p4);
  V *colv = reinterpret_cast<V *>(data_col + col_base);
  for (int c = lane; c < CinV; c += WARP_SIZE) {
    const V a1 = im1[c], a2 = im2[c], a3 = im3[c], a4 = im4[c];
    V o;
#pragma unroll
    for (int k = 0; k < VW; ++k)
      o.v[k] = (T)(cw1 * (float)a1.v[k] + cw2 * (float)a2.v[k] +
                   cw3 * (float)a3.v[k] + cw4 * (float)a4.v[k]);
    colv[c] = o;
  }
  for (int cc = CinV * VW + lane; cc < Cin; cc += WARP_SIZE) {  // remainder
    const float val = cw1 * (float)data_im[p1 + cc] + cw2 * (float)data_im[p2 + cc] +
                      cw3 * (float)data_im[p3 + cc] + cw4 * (float)data_im[p4 + cc];
    data_col[col_base + cc] = (T)val;
  }
}

// ---------------------------------------------------------------------------
// Fused F1-F5: offset conv (F1-F3) + bilinear gather (F4) + sigmoid modulation
// (F5) in ONE kernel. One warp per output pixel po=(b,oh,ow); lanes = channels.
//   Phase 1: regular offset conv -> Noff=3T raw outputs, via lanes-over-channels
//            partial sums + warp butterfly reduce (every lane holds all Noff).
//   Phase 2: for each tap, gather/modulate using the just-computed offset/mask
//            and write Col[po, t*Cin+ci]  (same vectorized path as the gather).
// Layout: w_off [Noff, T, Cin] (= torch [Noff,Cin,kH,kW].permute(0,2,3,1)),
//         b_off [Noff], data_col [M, T*Cin].  KH/KW are compile-time (3x3 only).
//
// acc[NOFF] must live in registers across both phases, so EVERY index into it is
// compile-time: the offset-conv output loop (NOFF) and the phase-2 tap loop (TT)
// are #pragma unroll'd. A runtime-indexed acc[] would spill to local memory and
// was ~50x slower.
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW>
__global__ void modulated_deformable_fused_im2col_nhwc_gpu_kernel(
    const T *__restrict__ data_im,    // [B, H, W, Cin]
    const T *__restrict__ w_off,      // [Noff, T, Cin]
    const float *__restrict__ b_off,  // [Noff]
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int kH_rt, const int kW_rt,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dil_h, const int dil_w,
    T *__restrict__ data_col) {       // [M, T*Cin]
  constexpr int TT = KH * KW;       // taps (compile-time)
  constexpr int NOFF = 3 * TT;      // offset-conv outputs (compile-time)

  const int WPB = blockDim.x >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int ow = blockIdx.x * WPB + (threadIdx.x >> 5);
  if (ow >= Wo) return;
  const int oh = blockIdx.y;
  const int b = blockIdx.z;
  const int m = (b * Ho + oh) * Wo + ow;
  const int Cin2 = Cin >> 1;
  const int TC = TT * Cin;

  // ---- Phase 1: offset conv -> acc[oc] (every lane gets the full reduced set) ----
  float acc[NOFF];
#pragma unroll
  for (int g = 0; g < NOFF; ++g) acc[g] = 0.f;

  for (int t = 0; t < TT; ++t) {
    const int ki = t / KW;
    const int kj = t - ki * KW;
    const int ih = oh * stride_h - pad_h + ki * dil_h;
    const int iw = ow * stride_w - pad_w + kj * dil_w;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;  // warp-uniform
    const Vec2<T> *xrow =
        reinterpret_cast<const Vec2<T> *>(data_im + ((b * H + ih) * W + iw) * Cin);
    const T *wt = w_off + t * Cin;  // weights for this tap, output g stride = TC
    for (int c = lane; c < Cin2; c += WARP_SIZE) {
      const Vec2<T> xv = xrow[c];
      const float x0 = (float)xv.x, x1 = (float)xv.y;
#pragma unroll
      for (int g = 0; g < NOFF; ++g) {
        const Vec2<T> wv =
            reinterpret_cast<const Vec2<T> *>(wt + g * TC)[c];
        acc[g] += (float)wv.x * x0 + (float)wv.y * x1;
      }
    }
  }
#pragma unroll
  for (int g = 0; g < NOFF; ++g) acc[g] = warp_all_sum(acc[g]) + b_off[g];

  // ---- Phase 2: gather + modulate, per tap, into Col[m, t*Cin+ci] ----
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    const int ki = t / KW;
    const int kj = t - ki * KW;
    const float off_h = acc[2 * t];
    const float off_w = acc[2 * t + 1];
    const float msk = sigmoidf(acc[2 * TT + t]);

    const float h_im = (float)(oh * stride_h - pad_h + ki * dil_h) + off_h;
    const float w_im = (float)(ow * stride_w - pad_w + kj * dil_w) + off_w;
    const int h_low = (int)floorf(h_im);
    const int w_low = (int)floorf(w_im);
    const float lh = h_im - (float)h_low, lw = w_im - (float)w_low;
    const float hh = 1.f - lh, hw = 1.f - lw;
    const int y0 = h_low, y1 = h_low + 1, x0 = w_low, x1 = w_low + 1;
    const bool g1 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W);
    const bool g2 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W);
    const bool g3 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W);
    const bool g4 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W);
    const int y0c = min(max(y0, 0), H - 1), y1c = min(max(y1, 0), H - 1);
    const int x0c = min(max(x0, 0), W - 1), x1c = min(max(x1, 0), W - 1);
    const float cw1 = hh * hw * (g1 ? msk : 0.f);
    const float cw2 = hh * lw * (g2 ? msk : 0.f);
    const float cw3 = lh * hw * (g3 ? msk : 0.f);
    const float cw4 = lh * lw * (g4 ? msk : 0.f);

    const int base_b = b * H * W * Cin;
    const Vec2<T> *im1 = reinterpret_cast<const Vec2<T> *>(data_im + base_b + (y0c * W + x0c) * Cin);
    const Vec2<T> *im2 = reinterpret_cast<const Vec2<T> *>(data_im + base_b + (y0c * W + x1c) * Cin);
    const Vec2<T> *im3 = reinterpret_cast<const Vec2<T> *>(data_im + base_b + (y1c * W + x0c) * Cin);
    const Vec2<T> *im4 = reinterpret_cast<const Vec2<T> *>(data_im + base_b + (y1c * W + x1c) * Cin);
    Vec2<T> *colv = reinterpret_cast<Vec2<T> *>(data_col + m * TC + t * Cin);
    for (int c = lane; c < Cin2; c += WARP_SIZE) {
      const Vec2<T> a1 = im1[c], a2 = im2[c], a3 = im3[c], a4 = im4[c];
      Vec2<T> o;
      o.x = (T)(cw1 * (float)a1.x + cw2 * (float)a2.x + cw3 * (float)a3.x + cw4 * (float)a4.x);
      o.y = (T)(cw1 * (float)a1.y + cw2 * (float)a2.y + cw3 * (float)a3.y + cw4 * (float)a4.y);
      colv[c] = o;
    }
  }
}

// ---------------------------------------------------------------------------
// Backward col2im (NHWC, dg==1): given dCol [M,T*Cin], X, offset, mask ->
//   gX   [B,H,W,Cin]   grad wrt the sampled input (4-corner atomic scatter)
//   gOff [B,Ho,Wo,2T]  grad wrt offsets (reduce over channels)
//   gMask[B,Ho,Wo,T]   grad wrt mask    (reduce over channels)
// One warp per (m,t); lanes = channels. grad_offset/grad_mask reduce over the
// warp's channels then ONE direct write (each (m,t) is owned by exactly one warp,
// so no atomic needed). grad_X must use atomicAdd (many (m,t) hit the same pixel).
// gX must be PRE-ZEROED; gOff/gMask are fully overwritten (no pre-zero needed).
// All grads accumulate in fp32. KH/KW compile-time (3x3).
//
// dcols = mask * S,  S = Σ w_corner(lh,lw)·g·v.   With dSf = dCol·mask:
//   dX[corner]   += dSf · w_corner            (valid corners only)
//   dOff_h        = Σ_ci dSf · (-hw·v1 - lw·v2 + hw·v3 + lw·v4)
//   dOff_w        = Σ_ci dSf · (-hh·v1 + hh·v2 - lh·v3 + lh·v4)
//   dMask         = Σ_ci dCol · S
// (per-corner gate folds into v=0 for invalid corners).
// ---------------------------------------------------------------------------
// WITH_GX=true -> also scatter grad_X (atomic); false -> grad_offset/grad_mask only
// (the cheap, atomic-free part; pair with the smem-tiled grad_X kernel below).
template <typename T, int KH, int KW, int VW, bool WITH_GX>
__global__ void modulated_deformable_col2im_nhwc_gpu_kernel(
    const T *__restrict__ dCol,       // [M, T*Cin]
    const T *__restrict__ data_im,    // [B, H, W, Cin]
    const T *__restrict__ offset,     // [B, Ho, Wo, 2T]
    const T *__restrict__ mask,       // [B, Ho, Wo, T]
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int kH_rt, const int kW_rt,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dil_h, const int dil_w,
    float *__restrict__ gX, float *__restrict__ gOff, float *__restrict__ gMask) {
  constexpr int TT = KH * KW;
  // occupancy-throttle experiment hook: a dummy dynamic-smem reservation set at
  // launch (DCN_DUMMY_SMEM) caps blocks/SM. Reference it under a never-true
  // (runtime) guard so the compiler keeps the reservation. No effect when 0.
  extern __shared__ char _occpad[];
  if (threadIdx.x == blockDim.x) _occpad[0] = 0;  // never executes
  const int WPB = blockDim.x >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int ow = blockIdx.x * WPB + (threadIdx.x >> 5);
  if (ow >= Wo) return;
  const int oh = blockIdx.y;
  const int b = blockIdx.z / TT;
  const int t = blockIdx.z - b * TT;
  const int ki = t / KW, kj = t - ki * KW;
  const int m = (b * Ho + oh) * Wo + ow;

  const float off_h = (float)offset[m * (2 * TT) + 2 * t];
  const float off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
  const float msk = (float)mask[m * TT + t];
  const float h_im = (float)(oh * stride_h - pad_h + ki * dil_h) + off_h;
  const float w_im = (float)(ow * stride_w - pad_w + kj * dil_w) + off_w;
  const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
  const float lh = h_im - (float)hl, lw = w_im - (float)wl;
  const float hh = 1.f - lh, hw = 1.f - lw;
  const int y0 = hl, y1 = hl + 1, x0 = wl, x1 = wl + 1;
  const bool g1 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W);
  const bool g2 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W);
  const bool g3 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W);
  const bool g4 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W);
  const int y0c = min(max(y0, 0), H - 1), y1c = min(max(y1, 0), H - 1);
  const int x0c = min(max(x0, 0), W - 1), x1c = min(max(x1, 0), W - 1);
  const int base_b = b * H * W * Cin;
  const int p1 = base_b + (y0c * W + x0c) * Cin, p2 = base_b + (y0c * W + x1c) * Cin;
  const int p3 = base_b + (y1c * W + x0c) * Cin, p4 = base_b + (y1c * W + x1c) * Cin;
  const int col_base = m * (TT * Cin) + t * Cin;
  const float w00 = hh * hw, w01 = hh * lw, w10 = lh * hw, w11 = lh * lw;

  float a_h = 0.f, a_w = 0.f, a_m = 0.f;
  // Vectorized: VW channels/lane. Reads (v1..v4, dc) coalesce as wide loads; the
  // grad_X scatter stays per-element (atomics). Reduction accumulates all VW.
  using V = VecN<T, VW>;
  const int CinV = Cin / VW;
  const V *im1 = reinterpret_cast<const V *>(data_im + p1);
  const V *im2 = reinterpret_cast<const V *>(data_im + p2);
  const V *im3 = reinterpret_cast<const V *>(data_im + p3);
  const V *im4 = reinterpret_cast<const V *>(data_im + p4);
  const V *dcv = reinterpret_cast<const V *>(dCol + col_base);
  // NOTE: explicit 2-stage software prefetch here gave ZERO speedup (4245 vs
  // 4239us) — the compiler already schedules independent loads ahead, and the
  // real stall is the LSU queue clogged by the scalar atomics (3.7x the reads),
  // which delays read ISSUE; prefetch can't bypass that. So: plain loop.
  for (int c = lane; c < CinV; c += WARP_SIZE) {
    const V A1 = im1[c], A2 = im2[c], A3 = im3[c], A4 = im4[c], DC = dcv[c];
#pragma unroll
    for (int k = 0; k < VW; ++k) {
      const float v1 = g1 ? (float)A1.v[k] : 0.f;
      const float v2 = g2 ? (float)A2.v[k] : 0.f;
      const float v3 = g3 ? (float)A3.v[k] : 0.f;
      const float v4 = g4 ? (float)A4.v[k] : 0.f;
      const float dc = (float)DC.v[k];
      const float dSf = dc * msk;
      const int cc = c * VW + k;
      if constexpr (WITH_GX) {
        if (g1) atomicAdd(&gX[p1 + cc], dSf * w00);
        if (g2) atomicAdd(&gX[p2 + cc], dSf * w01);
        if (g3) atomicAdd(&gX[p3 + cc], dSf * w10);
        if (g4) atomicAdd(&gX[p4 + cc], dSf * w11);
      }
      a_h += dSf * (-hw * v1 - lw * v2 + hw * v3 + lw * v4);
      a_w += dSf * (-hh * v1 + hh * v2 - lh * v3 + lh * v4);
      a_m += dc * (w00 * v1 + w01 * v2 + w10 * v3 + w11 * v4);
    }
  }
  for (int cc = CinV * VW + lane; cc < Cin; cc += WARP_SIZE) {  // remainder
    const float v1 = g1 ? (float)data_im[p1 + cc] : 0.f;
    const float v2 = g2 ? (float)data_im[p2 + cc] : 0.f;
    const float v3 = g3 ? (float)data_im[p3 + cc] : 0.f;
    const float v4 = g4 ? (float)data_im[p4 + cc] : 0.f;
    const float dc = (float)dCol[col_base + cc];
    const float dSf = dc * msk;
    if constexpr (WITH_GX) {
      if (g1) atomicAdd(&gX[p1 + cc], dSf * w00);
      if (g2) atomicAdd(&gX[p2 + cc], dSf * w01);
      if (g3) atomicAdd(&gX[p3 + cc], dSf * w10);
      if (g4) atomicAdd(&gX[p4 + cc], dSf * w11);
    }
    a_h += dSf * (-hw * v1 - lw * v2 + hw * v3 + lw * v4);
    a_w += dSf * (-hh * v1 + hh * v2 - lh * v3 + lh * v4);
    a_m += dc * (w00 * v1 + w01 * v2 + w10 * v3 + w11 * v4);
  }
  a_h = warp_all_sum(a_h);
  a_w = warp_all_sum(a_w);
  a_m = warp_all_sum(a_m);
  if (lane == 0) {
    gOff[m * (2 * TT) + 2 * t] = a_h;
    gOff[m * (2 * TT) + 2 * t + 1] = a_w;
    gMask[m * TT + t] = a_m;
  }
}

// ---------------------------------------------------------------------------
// Shared-memory-tiled grad_X (NHWC, dg==1, 3x3 stride-1 dil-1 — the BEVFormer
// case). Cuts global-atomic contention: a block owns an output tile (TH x TW)
// and a channel chunk (CC), scatters grad_X into a shared input window
// [SH x SW x CC] (channel-last) via fast SMEM atomics, then flushes the window
// to global with one atomic per non-zero cell. Out-of-window corners (large
// offsets) fall back to a direct global atomic. Mirrors the proven NCHW
// col2im_smem (2.93-3.28x) adapted to NHWC.  CC == warpSize so one warp covers
// the whole chunk per (output,tap).
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW, int CC, int TH, int TW, int MG>
__global__ void modulated_deformable_col2im_gradx_smem_nhwc_gpu_kernel(
    const T *__restrict__ dCol,    // [M, T*Cin]
    const T *__restrict__ offset,  // [B, Ho, Wo, 2T]
    const T *__restrict__ mask,    // [B, Ho, Wo, T]
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int pad_h, const int pad_w,
    const int NCHUNK, float *__restrict__ gX) {
  constexpr int TT = KH * KW;
  constexpr int SH = TH + 2 * MG;
  constexpr int SW = TW + 2 * MG;
  constexpr int SMEM_N = SH * SW * CC;
  extern __shared__ float smem[];

  const int bz = blockIdx.z;
  const int b = bz / NCHUNK;
  const int c0 = (bz - b * NCHUNK) * CC;
  const int th0 = blockIdx.y * TH, tw0 = blockIdx.x * TW;
  const int in_h0 = th0 - pad_h - MG;       // stride 1
  const int in_w0 = tw0 - pad_w - MG;

  for (int i = threadIdx.x; i < SMEM_N; i += blockDim.x) smem[i] = 0.f;
  __syncthreads();

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  const int n_ot = TH * TW * TT;
  for (int ot = warp; ot < n_ot; ot += nwarp) {
    const int lp = ot / TT, t = ot - lp * TT;
    const int lh = lp / TW, lw = lp - lh * TW;
    const int h_out = th0 + lh, w_out = tw0 + lw;
    if (h_out >= Ho || w_out >= Wo) continue;       // warp-uniform
    const int m = (b * Ho + h_out) * Wo + w_out;
    const int ki = t / KW, kj = t - ki * KW;
    const float off_h = (float)offset[m * (2 * TT) + 2 * t];
    const float off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
    const float msk = (float)mask[m * TT + t];
    const float h_im = (float)(h_out - pad_h + ki) + off_h;   // stride1 dil1
    const float w_im = (float)(w_out - pad_w + kj) + off_w;
    const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
    const float lhh = h_im - hl, lww = w_im - wl, hh = 1.f - lhh, hw = 1.f - lww;
    const float cwt[4] = {hh * hw, hh * lww, lhh * hw, lhh * lww};
    const int cy[4] = {hl, hl, hl + 1, hl + 1};
    const int cx[4] = {wl, wl + 1, wl, wl + 1};
    // geometry computed once per (output,tap); loop the chunk's CC channels in
    // 32-lane sub-groups so CC can exceed warpSize (fewer chunks -> less redundancy).
    const int dcol_row = m * (TT * Cin) + t * Cin;
#pragma unroll
    for (int cb = 0; cb < CC; cb += WARP_SIZE) {
      const int cl = cb + lane;
      const int cc = c0 + cl;
      const float dSf = (float)dCol[dcol_row + cc] * msk;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int yy = cy[q], xx = cx[q];
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;   // warp-uniform
        const int sh = yy - in_h0, sw = xx - in_w0;
        const float val = cwt[q] * dSf;
        if (sh >= 0 && sh < SH && sw >= 0 && sw < SW)
          atomicAdd(&smem[(sh * SW + sw) * CC + cl], val);
        else
          atomicAdd(&gX[((b * H + yy) * W + xx) * Cin + cc], val);
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < SMEM_N; i += blockDim.x) {
    const float val = smem[i];
    if (val == 0.f) continue;
    const int cl = i % CC;
    const int sp = i / CC;
    const int sh = sp / SW, sw = sp - sh * SW;
    const int gy = in_h0 + sh, gx = in_w0 + sw;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W)
      atomicAdd(&gX[((b * H + gy) * W + gx) * Cin + c0 + cl], val);
  }
}

// ---------------------------------------------------------------------------
// FULLY-FUSED B4-B9 with smem-tiled grad_X (one kernel, dg==1, 3x3 s1 d1).
// = gradx_smem (grad_X scatter -> smem window -> flush) PLUS grad_offset/grad_mask
// (read 4 X corners, reduce over the chunk's channels, atomic-accumulate the
// per-chunk PARTIAL to global gOff/gMask — small tensors, ~NCHUNK-way, cheap).
// So intermediate grad_X atomics only hit smem; only the flush + the tiny
// off/mask partials touch gmem.  gX, gOff, gMask PRE-ZEROED.
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW, int CC, int TH, int TW, int MG>
__global__ void modulated_deformable_col2im_fused_smem_nhwc_gpu_kernel(
    const T *__restrict__ dCol, const T *__restrict__ data_im,
    const T *__restrict__ offset, const T *__restrict__ mask,
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int pad_h, const int pad_w,
    const int NCHUNK, float *__restrict__ gX, float *__restrict__ gOff,
    float *__restrict__ gMask) {
  constexpr int TT = KH * KW;
  constexpr int SH = TH + 2 * MG;
  constexpr int SW = TW + 2 * MG;
  constexpr int SMEM_N = SH * SW * CC;
  extern __shared__ float smem[];

  const int bz = blockIdx.z;
  const int b = bz / NCHUNK;
  const int c0 = (bz - b * NCHUNK) * CC;
  const int th0 = blockIdx.y * TH, tw0 = blockIdx.x * TW;
  const int in_h0 = th0 - pad_h - MG, in_w0 = tw0 - pad_w - MG;

  for (int i = threadIdx.x; i < SMEM_N; i += blockDim.x) smem[i] = 0.f;
  __syncthreads();

  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  const int base_b = b * H * W * Cin;
  const int n_ot = TH * TW * TT;
  for (int ot = warp; ot < n_ot; ot += nwarp) {
    const int lp = ot / TT, t = ot - lp * TT;
    const int lh = lp / TW, lw = lp - lh * TW;
    const int h_out = th0 + lh, w_out = tw0 + lw;
    if (h_out >= Ho || w_out >= Wo) continue;
    const int m = (b * Ho + h_out) * Wo + w_out;
    const int ki = t / KW, kj = t - ki * KW;
    const float off_h = (float)offset[m * (2 * TT) + 2 * t];
    const float off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
    const float msk = (float)mask[m * TT + t];
    const float h_im = (float)(h_out - pad_h + ki) + off_h;
    const float w_im = (float)(w_out - pad_w + kj) + off_w;
    const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
    const float lhh = h_im - hl, lww = w_im - wl, hh = 1.f - lhh, hw = 1.f - lww;
    const float cwt[4] = {hh * hw, hh * lww, lhh * hw, lhh * lww};
    const float ahc[4] = {-hw, -lww, hw, lww};   // dS/d(off_h) coeffs per corner
    const float awc[4] = {-hh, hh, -lhh, lhh};   // dS/d(off_w) coeffs per corner
    const int cy[4] = {hl, hl, hl + 1, hl + 1};
    const int cx[4] = {wl, wl + 1, wl, wl + 1};
    const int dcol_row = m * (TT * Cin) + t * Cin;

    float a_h = 0.f, a_w = 0.f, a_m = 0.f;
#pragma unroll
    for (int cb = 0; cb < CC; cb += WARP_SIZE) {
      const int cc = c0 + cb + lane;
      const float dc = (float)dCol[dcol_row + cc];
      const float dSf = dc * msk;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int yy = cy[q], xx = cx[q];
        const bool inb = (yy >= 0 && yy < H && xx >= 0 && xx < W);
        // grad_X scatter (smem window, or gmem fallback out-of-window)
        if (inb) {
          const int sh = yy - in_h0, sw = xx - in_w0;
          const float val = cwt[q] * dSf;
          if (sh >= 0 && sh < SH && sw >= 0 && sw < SW)
            atomicAdd(&smem[(sh * SW + sw) * CC + cb + lane], val);
          else
            atomicAdd(&gX[((b * H + yy) * W + xx) * Cin + cc], val);
        }
        // grad_off / grad_mask need the corner X value (0 if out of bounds)
        const float v = inb ? (float)data_im[base_b + (yy * W + xx) * Cin + cc] : 0.f;
        a_h += dSf * ahc[q] * v;
        a_w += dSf * awc[q] * v;
        a_m += dc * cwt[q] * v;
      }
    }
    // reduce the chunk's channels, atomic-accumulate the partial to gOff/gMask
    a_h = warp_all_sum(a_h);
    a_w = warp_all_sum(a_w);
    a_m = warp_all_sum(a_m);
    if (lane == 0) {
      atomicAdd(&gOff[m * (2 * TT) + 2 * t], a_h);
      atomicAdd(&gOff[m * (2 * TT) + 2 * t + 1], a_w);
      atomicAdd(&gMask[m * TT + t], a_m);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < SMEM_N; i += blockDim.x) {
    const float val = smem[i];
    if (val == 0.f) continue;
    const int cl = i % CC, sp = i / CC;
    const int sh = sp / SW, sw = sp - sh * SW;
    const int gy = in_h0 + sh, gx = in_w0 + sw;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W)
      atomicAdd(&gX[((b * H + gy) * W + gx) * Cin + c0 + cl], val);
  }
}

// ---------------------------------------------------------------------------
// PER-ELEMENT col2im (TileLang-style): ONE thread per (m,t,ci). block = Cin
// (threads = ci), grid = (Wo, Ho, B*T) so (b,oh,ow,t) come from blockIdx (no
// runtime division — TileLang pays a strength-reduced div, we pay none). Each
// thread recomputes the full geometry: that "redundant" compute is exactly what
// HIDES the grad_X atomic latency (compute-bound, like TileLang's 77.5%/2.32ms),
// whereas amortizing geometry across channels (great for the fwd gather) exposes
// the atomics and is slower here. grad_off/mask: warp-reduce 32 ci + 1 atomic/warp.
// gX/gOff/gMask PRE-ZEROED. 3x3 s1 d1.
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW, bool FUSED_Y>
__global__ void modulated_deformable_col2im_perelem_nhwc_gpu_kernel(
    const T *__restrict__ dCol, const T *__restrict__ data_im,
    const T *__restrict__ offset, const T *__restrict__ mask,  // offset==y[3T] if FUSED_Y
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int pad_h, const int pad_w,
    float *__restrict__ gX, float *__restrict__ gOff, float *__restrict__ gMask) {
  constexpr int TT = KH * KW;
  const int ci = threadIdx.x;
  if (ci >= Cin) return;
  const int ow = blockIdx.x, oh = blockIdx.y;
  const int b = blockIdx.z / TT;
  const int t = blockIdx.z - b * TT;
  const int ki = t / KW, kj = t - ki * KW;
  const int m = (b * Ho + oh) * Wo + ow;

  float off_h, off_w, msk;
  if constexpr (FUSED_Y) {  // offset == raw conv output y [B,Ho,Wo,3T]
    const int yb = m * (3 * TT);
    off_h = (float)offset[yb + 2 * t];
    off_w = (float)offset[yb + 2 * t + 1];
    msk = sigmoidf((float)offset[yb + 2 * TT + t]);
  } else {
    off_h = (float)offset[m * (2 * TT) + 2 * t];
    off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
    msk = (float)mask[m * TT + t];
  }
  const float h_im = (float)(oh - pad_h + ki) + off_h;   // stride1 dil1
  const float w_im = (float)(ow - pad_w + kj) + off_w;
  const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
  const float lh = h_im - hl, lw = w_im - wl, hh = 1.f - lh, hw = 1.f - lw;
  const int y0 = hl, y1 = hl + 1, x0 = wl, x1 = wl + 1;
  const bool g1 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W);
  const bool g2 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W);
  const bool g3 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W);
  const bool g4 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W);
  const int y0c = min(max(y0, 0), H - 1), y1c = min(max(y1, 0), H - 1);
  const int x0c = min(max(x0, 0), W - 1), x1c = min(max(x1, 0), W - 1);
  const int base_b = b * H * W * Cin;
  const int p1 = base_b + (y0c * W + x0c) * Cin + ci;
  const int p2 = base_b + (y0c * W + x1c) * Cin + ci;
  const int p3 = base_b + (y1c * W + x0c) * Cin + ci;
  const int p4 = base_b + (y1c * W + x1c) * Cin + ci;
  const float v1 = g1 ? (float)data_im[p1] : 0.f;
  const float v2 = g2 ? (float)data_im[p2] : 0.f;
  const float v3 = g3 ? (float)data_im[p3] : 0.f;
  const float v4 = g4 ? (float)data_im[p4] : 0.f;
  const float dc = (float)dCol[m * (TT * Cin) + t * Cin + ci];
  const float dSf = dc * msk;
  const float w00 = hh * hw, w01 = hh * lw, w10 = lh * hw, w11 = lh * lw;

  if (g1) atomicAdd(&gX[p1], dSf * w00);
  if (g2) atomicAdd(&gX[p2], dSf * w01);
  if (g3) atomicAdd(&gX[p3], dSf * w10);
  if (g4) atomicAdd(&gX[p4], dSf * w11);

  float a_h = dSf * (-hw * v1 - lw * v2 + hw * v3 + lw * v4);
  float a_w = dSf * (-hh * v1 + hh * v2 - lh * v3 + lh * v4);
  float a_m = dc * (w00 * v1 + w01 * v2 + w10 * v3 + w11 * v4);
  a_h = warp_all_sum(a_h);
  a_w = warp_all_sum(a_w);
  a_m = warp_all_sum(a_m);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    atomicAdd(&gOff[m * (2 * TT) + 2 * t], a_h);
    atomicAdd(&gOff[m * (2 * TT) + 2 * t + 1], a_w);
    // FUSED sigmoid-backward: gMask out = grad wrt the PRE-sigmoid mask
    // (= a_m * sig*(1-sig)); msk is the post-sigmoid value, constant over the warp.
    atomicAdd(&gMask[m * TT + t], a_m * msk * (1.f - msk));
  }
}

// ---------------------------------------------------------------------------
// PER-ELEMENT col2im, smem-shared geometry variant: thread 0 computes the
// per-(m,t) geometry (corner base addrs, g-gated weights/coeffs) ONCE into smem;
// all Cin threads read it (vs recomputing 8x/block). Tests whether the geometry
// redundancy had slack (-> faster) or was load-bearing for hiding atomics (-> slower).
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW, bool FUSED_Y>
__global__ void modulated_deformable_col2im_perelem_smem_nhwc_gpu_kernel(
    const T *__restrict__ dCol, const T *__restrict__ data_im,
    const T *__restrict__ offset, const T *__restrict__ mask,  // offset==y[3T] if FUSED_Y
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int pad_h, const int pad_w,
    float *__restrict__ gX, float *__restrict__ gOff, float *__restrict__ gMask) {
  constexpr int TT = KH * KW;
  __shared__ int s_p[4];                 // clamped corner base addr (channel 0)
  __shared__ float s_sw[4], s_ahg[4], s_awg[4], s_msk;
  __shared__ int s_dcol, s_off2t, s_mskt;
  const int ow = blockIdx.x, oh = blockIdx.y;
  const int b = blockIdx.z / TT, t = blockIdx.z - b * TT;

  if (threadIdx.x == 0) {
    const int ki = t / KW, kj = t - ki * KW;
    const int m = (b * Ho + oh) * Wo + ow;
    float off_h, off_w;
    if constexpr (FUSED_Y) {  // offset == raw conv output y [B,Ho,Wo,3T]
      const int yb = m * (3 * TT);
      off_h = (float)offset[yb + 2 * t];
      off_w = (float)offset[yb + 2 * t + 1];
      s_msk = sigmoidf((float)offset[yb + 2 * TT + t]);
    } else {
      off_h = (float)offset[m * (2 * TT) + 2 * t];
      off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
      s_msk = (float)mask[m * TT + t];
    }
    const float h_im = (float)(oh - pad_h + ki) + off_h;
    const float w_im = (float)(ow - pad_w + kj) + off_w;
    const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
    const float lh = h_im - hl, lw = w_im - wl, hh = 1.f - lh, hw = 1.f - lw;
    const int yq[4] = {hl, hl, hl + 1, hl + 1};
    const int xq[4] = {wl, wl + 1, wl, wl + 1};
    const float cwt[4] = {hh * hw, hh * lw, lh * hw, lh * lw};
    const float ahc[4] = {-hw, -lw, hw, lw};
    const float awc[4] = {-hh, hh, -lh, lh};
    const int base_b = b * H * W * Cin;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int yc = min(max(yq[q], 0), H - 1), xc = min(max(xq[q], 0), W - 1);
      const float g = (yq[q] >= 0 && yq[q] < H && xq[q] >= 0 && xq[q] < W) ? 1.f : 0.f;
      s_p[q] = base_b + (yc * W + xc) * Cin;
      s_sw[q] = g * cwt[q];
      s_ahg[q] = g * ahc[q];
      s_awg[q] = g * awc[q];
    }
    s_dcol = m * (TT * Cin) + t * Cin;
    s_off2t = m * (2 * TT) + 2 * t;
    s_mskt = m * TT + t;
  }
  __syncthreads();

  const int ci = threadIdx.x;
  if (ci >= Cin) return;
  const float x0 = (float)data_im[s_p[0] + ci], x1 = (float)data_im[s_p[1] + ci];
  const float x2 = (float)data_im[s_p[2] + ci], x3 = (float)data_im[s_p[3] + ci];
  const float dc = (float)dCol[s_dcol + ci];
  const float dSf = dc * s_msk;
  atomicAdd(&gX[s_p[0] + ci], s_sw[0] * dSf);  // unconditional (sw=0 if OOB)
  atomicAdd(&gX[s_p[1] + ci], s_sw[1] * dSf);
  atomicAdd(&gX[s_p[2] + ci], s_sw[2] * dSf);
  atomicAdd(&gX[s_p[3] + ci], s_sw[3] * dSf);
  float a_h = dSf * (s_ahg[0] * x0 + s_ahg[1] * x1 + s_ahg[2] * x2 + s_ahg[3] * x3);
  float a_w = dSf * (s_awg[0] * x0 + s_awg[1] * x1 + s_awg[2] * x2 + s_awg[3] * x3);
  float a_m = dc * (s_sw[0] * x0 + s_sw[1] * x1 + s_sw[2] * x2 + s_sw[3] * x3);
  a_h = warp_all_sum(a_h);
  a_w = warp_all_sum(a_w);
  a_m = warp_all_sum(a_m);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    atomicAdd(&gOff[s_off2t], a_h);
    atomicAdd(&gOff[s_off2t + 1], a_w);
    atomicAdd(&gMask[s_mskt], a_m * s_msk * (1.f - s_msk));  // fused sigmoid-bwd
  }
}

// ---------------------------------------------------------------------------
// ATOMIC-FREE grad_X via gather (owner-computes). Each warp owns one input pixel
// (b,y,x) + a 32-channel group; it enumerates candidate outputs within a +-WIN
// window of the position whose regular tap would land on (y,x), checks (exactly,
// per candidate) whether (y,x) is one of that output's 4 bilinear corners, and
// accumulates dCol*mask*corner_weight. grad_X[b,y,x,ci] is written by exactly
// one warp -> NO atomics, direct store.
//   offset/mask are channel-independent -> read once per candidate (warp
//   broadcast); dCol[...,cc] is coalesced across lanes.
// CAVEAT: a contributing output with |offset| > WIN is missed (approximate for
// large offsets). Exact when offsets stay within +-WIN. 3x3 stride-1 dil-1.
// ---------------------------------------------------------------------------
template <typename T, int KH, int KW, int WIN>
__global__ void modulated_deformable_col2im_gradx_gather_nhwc_gpu_kernel(
    const T *__restrict__ dCol,    // [M, T*Cin]
    const T *__restrict__ offset,  // [B, Ho, Wo, 2T]
    const T *__restrict__ mask,    // [B, Ho, Wo, T]
    const int B, const int Cin, const int H, const int W,
    const int Ho, const int Wo, const int pad_h, const int pad_w,
    const int NCG, float *__restrict__ gX) {
  constexpr int TT = KH * KW;
  const int WPB = blockDim.x >> 5;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int x = blockIdx.x * WPB + (threadIdx.x >> 5);
  if (x >= W) return;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
  const int b = z / NCG;
  const int cc = (z - b * NCG) * WARP_SIZE + lane;

  float acc = 0.f;
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    const int ki = t / KW, kj = t - ki * KW;
    const int ohc = y + pad_h - ki, owc = x + pad_w - kj;  // stride1 dil1, off=0 center
    for (int doh = -WIN; doh <= WIN; ++doh) {
      const int oh = ohc + doh;
      if ((unsigned)oh >= (unsigned)Ho) continue;
      for (int dow = -WIN; dow <= WIN; ++dow) {
        const int ow = owc + dow;
        if ((unsigned)ow >= (unsigned)Wo) continue;
        const int m = (b * Ho + oh) * Wo + ow;
        const float off_h = (float)offset[m * (2 * TT) + 2 * t];  // warp-broadcast
        const float off_w = (float)offset[m * (2 * TT) + 2 * t + 1];
        const float h_im = (float)(oh - pad_h + ki) + off_h;
        const float w_im = (float)(ow - pad_w + kj) + off_w;
        const int hl = (int)floorf(h_im), wl = (int)floorf(w_im);
        if ((y == hl || y == hl + 1) && (x == wl || x == wl + 1)) {
          const float lh = h_im - hl, lw = w_im - wl;
          const float wy = (y == hl) ? (1.f - lh) : lh;
          const float wx = (x == wl) ? (1.f - lw) : lw;
          acc += (float)dCol[m * (TT * Cin) + t * Cin + cc] *
                 (float)mask[m * TT + t] * wy * wx;
        }
      }
    }
  }
  gX[((b * H + y) * W + x) * Cin + cc] = acc;  // direct write, no atomic
}

}  // namespace dcn_nhwc

#endif  // DCN_FUSED_NHWC_KERNEL_CUH
