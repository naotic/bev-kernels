#pragma once

#include "msda_macros.cuh"
#include <cstdint>
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"

template <typename scalar_t>
__device__ inline scalar_t msda_im2col_bilinear(
    const scalar_t *bottom_data, int height, int width, int nheads, int channels,
    scalar_t h, scalar_t w, int m, int c) {
  const int h_low = floorf(static_cast<float>(h));
  const int w_low = floorf(static_cast<float>(w));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - static_cast<scalar_t>(h_low);
  const scalar_t lw = w - static_cast<scalar_t>(w_low);
  const scalar_t hh = static_cast<scalar_t>(1) - lh;
  const scalar_t hw = static_cast<scalar_t>(1) - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <int NUM_LEVELS, int NUM_POINTS>
__global__ void ms_deform_im2col_spec_forward_f32_warp4_shm_ptr_kernel(
    const float *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, int spatial_size, int num_query,
    int batch_size, float *data_col) {
  constexpr int kHeads = 8;
  constexpr int kChannels = 32;
  const int qid_stride = kHeads * kChannels;

  const int warp_id = threadIdx.x >> 5;  // 0..3
  const int lane = threadIdx.x & 31;     // 0..31
  const int total = batch_size * num_query * kHeads;
  const int unit = blockIdx.x * 4 + warp_id;
  if (unit >= total) return;

  const int c = lane;

  // shared: per warp store loc_w/loc_h/weight + warp-geometry scalars
  // Geometry layout (40 bytes / warp):
  //   5 floats: {point_weight, w1..w4}
  //   5 ints:   {h_low,w_low,h_high,w_high, do_compute}
  struct WarpGeomI32 {
    float point_weight;
    float w1, w2, w3, w4;
    int h_low, w_low, h_high, w_high;
    int do_compute;
  };
  static_assert(sizeof(WarpGeomI32) == 40, "Unexpected WarpGeomI32 size/padding; update shmem layout");

  extern __shared__ unsigned char shm_raw[];
  constexpr size_t kLocElems = static_cast<size_t>(NUM_LEVELS * NUM_POINTS * 3);
  constexpr size_t kLocBytes = kLocElems * sizeof(float);
  // SpecA is small (NUM_LEVELS=1, NUM_POINTS=4). For this case the overhead of warp-level
  // geometry precompute outweighs its benefit, so we keep the original per-lane bilinear path.
  constexpr bool kEnableGeomPrecompute = !(NUM_LEVELS == 1 && NUM_POINTS == 4);
  constexpr size_t kGeomBytes = kEnableGeomPrecompute ? sizeof(WarpGeomI32) : 0;

  const size_t warp_base_bytes =
      static_cast<size_t>(warp_id) * (kLocBytes + kGeomBytes);
  float* warp_buf = reinterpret_cast<float*>(shm_raw + warp_base_bytes);
  WarpGeomI32* warp_geom =
      reinterpret_cast<WarpGeomI32*>(shm_raw + warp_base_bytes + kLocBytes);

  float* warp_loc_w = warp_buf;
  float* warp_loc_h = warp_buf + (NUM_LEVELS * NUM_POINTS);
  float* warp_w = warp_buf + (NUM_LEVELS * NUM_POINTS * 2);

  const int b_col = unit / (num_query * kHeads);
  const int rem = unit - b_col * num_query * kHeads;
  const int q_col = rem / kHeads;
  const int m_col = rem - q_col * kHeads;

  const int sampling_index = ((b_col * num_query + q_col) * kHeads + m_col);
  const int base_weight_ptr = sampling_index * (NUM_LEVELS * NUM_POINTS);
  const int base_loc_ptr = base_weight_ptr << 1;

  if (lane < (NUM_LEVELS * NUM_POINTS)) {
    const int i = lane;
    warp_loc_w[i] = data_sampling_loc[base_loc_ptr + i * 2];
    warp_loc_h[i] = data_sampling_loc[base_loc_ptr + i * 2 + 1];
    warp_w[i] = data_attn_weight[base_weight_ptr + i];
  }
  __syncwarp();

  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    // per-level precompute: spatial_h/w and value base pointer
    int spatial_h_arr[NUM_LEVELS];
    int spatial_w_arr[NUM_LEVELS];
    const float* value_ptr_arr[NUM_LEVELS];
#pragma unroll
    for (int l = 0; l < NUM_LEVELS; ++l) {
      const int ph = l << 1;
      const int spatial_h = static_cast<int>(data_spatial_shapes[ph]);
      const int spatial_w = static_cast<int>(data_spatial_shapes[ph + 1]);
      const int64_t level_start_id = data_level_start_index[l];
      spatial_h_arr[l] = spatial_h;
      spatial_w_arr[l] = spatial_w;
      value_ptr_arr[l] =
          data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
    }

    float acc = 0.f;
    int idx = 0;
#pragma unroll
    for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
      const int spatial_h = spatial_h_arr[l_col];
      const int spatial_w = spatial_w_arr[l_col];
      const float* data_value_ptr = value_ptr_arr[l_col];
#pragma unroll
      for (int p_col = 0; p_col < NUM_POINTS; ++p_col) {
      if constexpr (kEnableGeomPrecompute) {
        // When num_query is small (SpecB-ish: 2048), geometry precompute overhead
        // can dominate. Use precompute only when num_query is large enough.
        // SpecB-real 的当前 perf 回归表明：几何预计算路径的开销在本实现下
        // 仍可能盖过收益。因此先把门限调高，让 SpecB-ish/SpecB-real 都走
        // 原始 per-lane bilinear（后续可以再针对 SpecB-real 做更细的优化）。
        constexpr int kPrecomputeMinNumQuery = 20000;
        const bool do_precompute = (num_query >= kPrecomputeMinNumQuery);

        if (do_precompute) {
          // warp-level bilinear geometry precompute (shared reuse):
          // lane0 computes floor/bounds/weights once, writes scalar geometry into shared.
          // other lanes reuse these scalars and only load v1..v4 for their lane-specific channel `c`.
          if (lane == 0) {
            const float loc_w = warp_loc_w[idx];
            const float loc_h = warp_loc_h[idx];
            warp_geom->point_weight = warp_w[idx];  // point_weight

            const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
            const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;

            const bool in_range =
                (h_im > -1.f && w_im > -1.f &&
                 h_im < static_cast<float>(spatial_h) &&
                 w_im < static_cast<float>(spatial_w));

            if (in_range) {
              const int h_low = static_cast<int>(floorf(h_im));
              const int w_low = static_cast<int>(floorf(w_im));
              const int h_high = h_low + 1;
              const int w_high = w_low + 1;

              const float lh = h_im - static_cast<float>(h_low);
              const float lw = w_im - static_cast<float>(w_low);
              const float hh = 1.f - lh;
              const float hw = 1.f - lw;

              // w1..w4
              warp_geom->w1 = hh * hw;
              warp_geom->w2 = hh * lw;
              warp_geom->w3 = lh * hw;
              warp_geom->w4 = lh * lw;

              // indices
              warp_geom->h_low = h_low;
              warp_geom->w_low = w_low;
              warp_geom->h_high = h_high;
              warp_geom->w_high = w_high;
              warp_geom->do_compute = 1;
            } else {
              // In-range guard fails -> contribution is zero (same as original).
              warp_geom->w1 = warp_geom->w2 = warp_geom->w3 = warp_geom->w4 = 0.f;
              warp_geom->h_low =
                  warp_geom->w_low = warp_geom->h_high = warp_geom->w_high = 0;
              warp_geom->do_compute = 0;
            }
          }
          __syncwarp();

          const float point_weight = warp_geom->point_weight;
          const float w1 = warp_geom->w1;
          const float w2 = warp_geom->w2;
          const float w3 = warp_geom->w3;
          const float w4 = warp_geom->w4;
          const int h_low = warp_geom->h_low;
          const int w_low = warp_geom->w_low;
          const int h_high = warp_geom->h_high;
          const int w_high = warp_geom->w_high;
          const int do_compute = warp_geom->do_compute;

          if (do_compute) {
            float v1 = 0.f, v2 = 0.f, v3 = 0.f, v4 = 0.f;

            // shared computations per lane
            constexpr int kWStride = kHeads * kChannels;
            const int h_stride = spatial_w * kWStride;
            const int h_low_ptr_offset = h_low * h_stride;
            const int h_high_ptr_offset = h_high * h_stride;
            const int w_low_ptr_offset = w_low * kWStride;
            const int w_high_ptr_offset = w_high * kWStride;
            const int base_ptr_m = m_col * kChannels;

            const int ptr1_base =
                h_low_ptr_offset + w_low_ptr_offset + base_ptr_m;
            const int ptr2_base =
                h_low_ptr_offset + w_high_ptr_offset + base_ptr_m;
            const int ptr3_base =
                h_high_ptr_offset + w_low_ptr_offset + base_ptr_m;
            const int ptr4_base =
                h_high_ptr_offset + w_high_ptr_offset + base_ptr_m;

            // validity checks are uniform across warp (derived from shared indices).
            if (h_low >= 0 && w_low >= 0)
              v1 = data_value_ptr[ptr1_base + c];
            if (h_low >= 0 && w_high <= spatial_w - 1)
              v2 = data_value_ptr[ptr2_base + c];
            if (h_high <= spatial_h - 1 && w_low >= 0)
              v3 = data_value_ptr[ptr3_base + c];
            if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1)
              v4 = data_value_ptr[ptr4_base + c];

            acc +=
                (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) * point_weight;
          }
        } else {
          // Original per-lane bilinear path (used for SpecB-ish).
          const float loc_w = warp_loc_w[idx];
          const float loc_h = warp_loc_h[idx];
          const float weight = warp_w[idx];
          const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
          const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;
          if (h_im > -1.f && w_im > -1.f && h_im < spatial_h &&
              w_im < spatial_w) {
            acc +=
                msda_im2col_bilinear(data_value_ptr, spatial_h, spatial_w,
                                     kHeads, kChannels, h_im, w_im, m_col, c) *
                weight;
          }
        }
      } else {
        // Original per-lane bilinear path (used by SpecA only).
        const float loc_w = warp_loc_w[idx];
        const float loc_h = warp_loc_h[idx];
        const float weight = warp_w[idx];
        const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
        const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;
        if (h_im > -1.f && w_im > -1.f && h_im < spatial_h &&
            w_im < spatial_w) {
          acc += msda_im2col_bilinear(data_value_ptr, spatial_h, spatial_w,
                                      kHeads, kChannels, h_im, w_im, m_col,
                                      c) *
                 weight;
        }
      }
      idx += 1;
    }
  }

  const int out_idx =
      ((b_col * num_query + q_col) * kHeads + m_col) * kChannels + c;
  data_col[out_idx] = acc;
}

// ---------------------------------------------------------------------------
// fp16 (__half) forward specialization.
//
// Mirrors ms_deform_im2col_spec_forward_f32_warp4_shm_ptr_kernel exactly in
// geometry / gate logic (kHeads=8, kChannels=32, warp4, NUM_LEVELS/NUM_POINTS
// templated). Difference: value / sampling_loc / attn_weight are loaded as
// __half, immediately widened to float via __half2float, the bilinear +
// weighted-sum accumulation is done in FLOAT (acc is a float), and only the
// final result is narrowed back to __half on store. This satisfies the team's
// "fp16 deformable attn is unstable because of many sum ops" constraint: all
// summation happens in fp32.
//
// We deliberately use the plain per-lane bilinear path (no warp-geometry
// precompute) — that path is the one the real SpecB shape (Q=10000 < 20000)
// already takes in f32, so numerics line up and the code stays simple.
template <int NUM_LEVELS, int NUM_POINTS>
__global__ void ms_deform_im2col_spec_forward_f16_warp4_shm_ptr_kernel(
    const __half *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const __half *data_sampling_loc,
    const __half *data_attn_weight, int spatial_size, int num_query,
    int batch_size, __half *data_col) {
  constexpr int kHeads = 8;
  constexpr int kChannels = 32;
  const int qid_stride = kHeads * kChannels;

  const int warp_id = threadIdx.x >> 5;  // 0..3
  const int lane = threadIdx.x & 31;     // 0..31
  const int total = batch_size * num_query * kHeads;
  const int unit = blockIdx.x * 4 + warp_id;
  if (unit >= total) return;

  const int c = lane;

  // shared: per warp store loc_w/loc_h/weight as FLOAT (already widened).
  extern __shared__ unsigned char shm_raw[];
  constexpr size_t kLocElems = static_cast<size_t>(NUM_LEVELS * NUM_POINTS * 3);
  constexpr size_t kLocBytes = kLocElems * sizeof(float);
  const size_t warp_base_bytes = static_cast<size_t>(warp_id) * kLocBytes;
  float *warp_buf = reinterpret_cast<float *>(shm_raw + warp_base_bytes);

  float *warp_loc_w = warp_buf;
  float *warp_loc_h = warp_buf + (NUM_LEVELS * NUM_POINTS);
  float *warp_w = warp_buf + (NUM_LEVELS * NUM_POINTS * 2);

  const int b_col = unit / (num_query * kHeads);
  const int rem = unit - b_col * num_query * kHeads;
  const int q_col = rem / kHeads;
  const int m_col = rem - q_col * kHeads;

  const int sampling_index = ((b_col * num_query + q_col) * kHeads + m_col);
  const int base_weight_ptr = sampling_index * (NUM_LEVELS * NUM_POINTS);
  const int base_loc_ptr = base_weight_ptr << 1;

  if (lane < (NUM_LEVELS * NUM_POINTS)) {
    const int i = lane;
    warp_loc_w[i] = __half2float(data_sampling_loc[base_loc_ptr + i * 2]);
    warp_loc_h[i] = __half2float(data_sampling_loc[base_loc_ptr + i * 2 + 1]);
    warp_w[i] = __half2float(data_attn_weight[base_weight_ptr + i]);
  }
  __syncwarp();

  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

  int spatial_h_arr[NUM_LEVELS];
  int spatial_w_arr[NUM_LEVELS];
  const __half *value_ptr_arr[NUM_LEVELS];
#pragma unroll
  for (int l = 0; l < NUM_LEVELS; ++l) {
    const int ph = l << 1;
    const int spatial_h = static_cast<int>(data_spatial_shapes[ph]);
    const int spatial_w = static_cast<int>(data_spatial_shapes[ph + 1]);
    const int64_t level_start_id = data_level_start_index[l];
    spatial_h_arr[l] = spatial_h;
    spatial_w_arr[l] = spatial_w;
    value_ptr_arr[l] =
        data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
  }

  float acc = 0.f;
  int idx = 0;
#pragma unroll
  for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
    const int spatial_h = spatial_h_arr[l_col];
    const int spatial_w = spatial_w_arr[l_col];
    const __half *data_value_ptr = value_ptr_arr[l_col];
#pragma unroll
    for (int p_col = 0; p_col < NUM_POINTS; ++p_col) {
      const float loc_w = warp_loc_w[idx];
      const float loc_h = warp_loc_h[idx];
      const float weight = warp_w[idx];
      const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
      const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;
      if (h_im > -1.f && w_im > -1.f && h_im < spatial_h && w_im < spatial_w) {
        // Inlined bilinear gather in FLOAT (loads are __half -> __half2float).
        const int h_low = static_cast<int>(floorf(h_im));
        const int w_low = static_cast<int>(floorf(w_im));
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;

        const float lh = h_im - static_cast<float>(h_low);
        const float lw = w_im - static_cast<float>(w_low);
        const float hh = 1.f - lh;
        const float hw = 1.f - lw;

        constexpr int kWStride = kHeads * kChannels;
        const int h_stride = spatial_w * kWStride;
        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w_low * kWStride;
        const int w_high_ptr_offset = w_low_ptr_offset + kWStride;
        const int base_ptr_m = m_col * kChannels + c;

        float v1 = 0.f, v2 = 0.f, v3 = 0.f, v4 = 0.f;
        if (h_low >= 0 && w_low >= 0)
          v1 = __half2float(
              data_value_ptr[h_low_ptr_offset + w_low_ptr_offset + base_ptr_m]);
        if (h_low >= 0 && w_high <= spatial_w - 1)
          v2 = __half2float(
              data_value_ptr[h_low_ptr_offset + w_high_ptr_offset + base_ptr_m]);
        if (h_high <= spatial_h - 1 && w_low >= 0)
          v3 = __half2float(
              data_value_ptr[h_high_ptr_offset + w_low_ptr_offset + base_ptr_m]);
        if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1)
          v4 = __half2float(
              data_value_ptr[h_high_ptr_offset + w_high_ptr_offset + base_ptr_m]);

        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
        acc += (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) * weight;
      }
      idx += 1;
    }
  }

  const int out_idx =
      ((b_col * num_query + q_col) * kHeads + m_col) * kChannels + c;
  data_col[out_idx] = __float2half(acc);
}

#include <musa_runtime.h>

template <typename scalar_t>
__device__ void msda_col2im_bilinear_grad(
    const scalar_t *bottom_data, int height, int width, int nheads, int channels,
    scalar_t h, scalar_t w, int m, int c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *grad_value, scalar_t *grad_w_acc,
    scalar_t *grad_h_acc, scalar_t *grad_a_acc) {
  const int h_low = floorf(static_cast<float>(h));
  const int w_low = floorf(static_cast<float>(w));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - static_cast<scalar_t>(h_low);
  const scalar_t lw = w - static_cast<scalar_t>(w_low);
  const scalar_t hh = static_cast<scalar_t>(1) - lh;
  const scalar_t hw = static_cast<scalar_t>(1) - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_a_acc = top_grad * val;
  *grad_w_acc = static_cast<scalar_t>(width) * grad_w_weight * top_grad_value;
  *grad_h_acc = static_cast<scalar_t>(height) * grad_h_weight * top_grad_value;
}

// Precomputed version for backward bilinear grad:
// - h_im/w_im 的 floor/bounds/权重(w1..w4)由调用方提前计算
// - 本 helper 只做 corner load + grad 计算 + atomicAdd（数学与原 helper 对齐）
__device__ inline void msda_col2im_bilinear_grad_precomp_f32(
    const float *bottom_data, int height, int width, int channels,
    int h_low_ptr_offset, int h_high_ptr_offset, int w_low_ptr_offset,
    int w_high_ptr_offset, float lh, float lw, float hh, float hw,
    float w1, float w2, float w3, float w4, int valid1, int valid2,
    int valid3, int valid4, int m, int c, float top_grad,
    float attn_weight, float *grad_value, float *grad_w_acc,
    float *grad_h_acc, float *grad_a_acc) {
  const float top_grad_value = top_grad * attn_weight;
  float grad_h_weight = 0.f, grad_w_weight = 0.f;

  float v1 = 0.f;
  if (valid1) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + m * channels + c;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }

  float v2 = 0.f;
  if (valid2) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + m * channels + c;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }

  float v3 = 0.f;
  if (valid3) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + m * channels + c;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }

  float v4 = 0.f;
  if (valid4) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + m * channels + c;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_a_acc = top_grad * val;
  *grad_w_acc = static_cast<float>(width) * grad_w_weight * top_grad_value;
  *grad_h_acc = static_cast<float>(height) * grad_h_weight * top_grad_value;
}

__device__ inline void msda_warp_reduce_sum3_f32(float &a, float &b, float &c) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_down_sync(mask, a, offset);
    b += __shfl_down_sync(mask, b, offset);
    c += __shfl_down_sync(mask, c, offset);
  }
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_generic_kernel(
    const int n, const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *data_col, at::musa::FastDivmod fastdivmod_channels,
    at::musa::FastDivmod fastdivmod_num_heads,
    at::musa::FastDivmod fastdivmod_num_query) {
  MSDA_1D_KERNEL_LOOP(index, n) {
    uint32_t _temp = index;
    uint32_t c_col;
    uint32_t sampling_index;
    uint32_t m_col;
    fastdivmod_channels(sampling_index, c_col, _temp);
    uint32_t q_col;
    fastdivmod_num_heads(_temp, m_col, sampling_index);
    uint32_t b_col;
    fastdivmod_num_query(b_col, q_col, _temp);

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
      const int spatial_w =
          static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im =
            loc_h * static_cast<scalar_t>(spatial_h) -
            static_cast<scalar_t>(0.5);
        const scalar_t w_im =
            loc_w * static_cast<scalar_t>(spatial_w) -
            static_cast<scalar_t>(0.5);

        if (h_im > static_cast<scalar_t>(-1) && w_im > static_cast<scalar_t>(-1) &&
            h_im < static_cast<scalar_t>(spatial_h) &&
            w_im < static_cast<scalar_t>(spatial_w)) {
          col += msda_im2col_bilinear(data_value_ptr, spatial_h, spatial_w,
                                      num_heads, channels, h_im, w_im, m_col,
                                      c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t>
__device__ inline scalar_t msda_grad_col_load(
    const scalar_t *grad_col, int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int b, int q, int m, int c, bool contig, int channels, int num_heads,
    int num_query) {
  if (contig) {
    const int idx =
        ((b * num_query + q) * num_heads + m) * channels + c;
    return grad_col[idx];
  }
  return grad_col[b * s0 + q * s1 + m * s2 + c * s3];
}

__global__ void ms_deform_col2im_channels32_warp_reduce_f32_kernel(
    const int n, const float *grad_col, int64_t g_stride_b, int64_t g_stride_q,
    int64_t g_stride_m, int64_t g_stride_c, bool grad_col_contiguous,
    const float *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, const int batch_size, const int spatial_size,
    const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, float *grad_value,
    float *grad_sampling_loc, float *grad_attn_weight, 
    at::musa::FastDivmod fastdivmod_channels, at::musa::FastDivmod fastdivmod_num_heads, 
    at::musa::FastDivmod fastdivmod_num_query) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    return;

  uint32_t _temp = static_cast<uint32_t>(index);
  uint32_t c_col;
  uint32_t sampling_index;
  uint32_t m_col;
  uint32_t q_col;
  uint32_t b_col;
  fastdivmod_channels(sampling_index, c_col, _temp);
  fastdivmod_num_heads(_temp, m_col, sampling_index);
  fastdivmod_num_query(b_col, q_col, _temp);

  const float top_grad = msda_grad_col_load(
      grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c, b_col, q_col,
      m_col, c_col, grad_col_contiguous, channels, num_heads, num_query);

  int data_weight_ptr = sampling_index * num_levels * num_point;
  int data_loc_w_ptr = data_weight_ptr << 1;
  const int grad_sampling_ptr = data_weight_ptr;
  float *grad_sampling_ptr_base = grad_sampling_loc + (grad_sampling_ptr << 1);
  float *grad_attn_ptr_base = grad_attn_weight + grad_sampling_ptr;

  const int qid_stride = num_heads * channels;
  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

  const int lane = threadIdx.x & 31;
  // Per-warp slot so multiple warps per block don't stomp the geometry cache.
  // MSDA_COL2IM_MAX_WARPS bounds the static shared arrays; the launch block
  // size must satisfy blockDim.x/32 <= MSDA_COL2IM_MAX_WARPS.
  const int warp_id = threadIdx.x >> 5;

  // Warp-level bilinear geometry cache for backward.
  // Since all warp lanes share the same (h_im,w_im) for a given (level,point),
  // lane0 computes floor/bounds/weights once and all lanes reuse them.
  // Arrays are indexed by warp_id so each warp owns a private slot.
  __shared__ int sh_do_compute[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_valid1[MSDA_COL2IM_MAX_WARPS], sh_valid2[MSDA_COL2IM_MAX_WARPS],
      sh_valid3[MSDA_COL2IM_MAX_WARPS], sh_valid4[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_lh[MSDA_COL2IM_MAX_WARPS], sh_lw[MSDA_COL2IM_MAX_WARPS],
      sh_hh[MSDA_COL2IM_MAX_WARPS], sh_hw[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_w1[MSDA_COL2IM_MAX_WARPS], sh_w2[MSDA_COL2IM_MAX_WARPS],
      sh_w3[MSDA_COL2IM_MAX_WARPS], sh_w4[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_h_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_h_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_w_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_w_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];

  for (int l_col = 0; l_col < num_levels; ++l_col) {
    const int64_t level_start_id = data_level_start_index[l_col];
    const int spatial_h_ptr = l_col << 1;
    const int spatial_h = static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
    const int spatial_w =
        static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
    const int value_ptr_offset =
        data_value_ptr_init_offset + level_start_id * qid_stride;
    const float *data_value_ptr = data_value + value_ptr_offset;
    float *grad_value_ptr = grad_value + value_ptr_offset;

    for (int p_col = 0; p_col < num_point; ++p_col) {
      const float loc_w = data_sampling_loc[data_loc_w_ptr];
      const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
      const float weight = data_attn_weight[data_weight_ptr];

      const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
      const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;

      float gw = 0.f, gh = 0.f, ga = 0.f;

      if (lane == 0) {
        const bool in_range =
            (h_im > -1.f && w_im > -1.f && h_im < static_cast<float>(spatial_h) &&
             w_im < static_cast<float>(spatial_w));
        if (in_range) {
          const int h_low = floorf(h_im);
          const int w_low = floorf(w_im);
          const int h_high = h_low + 1;
          const int w_high = w_low + 1;

          sh_lh[warp_id] = h_im - static_cast<float>(h_low);
          sh_lw[warp_id] = w_im - static_cast<float>(w_low);
          sh_hh[warp_id] = 1.f - sh_lh[warp_id];
          sh_hw[warp_id] = 1.f - sh_lw[warp_id];

          sh_w1[warp_id] = sh_hh[warp_id] * sh_hw[warp_id];
          sh_w2[warp_id] = sh_hh[warp_id] * sh_lw[warp_id];
          sh_w3[warp_id] = sh_lh[warp_id] * sh_hw[warp_id];
          sh_w4[warp_id] = sh_lh[warp_id] * sh_lw[warp_id];

          sh_valid1[warp_id] = (h_low >= 0 && w_low >= 0);
          sh_valid2[warp_id] =
              (h_low >= 0 && w_high <= spatial_w - 1);
          sh_valid3[warp_id] =
              (h_high <= spatial_h - 1 && w_low >= 0);
          sh_valid4[warp_id] = (h_high <= spatial_h - 1 && w_high <= spatial_w - 1);

          const int w_stride = num_heads * channels;
          const int h_stride = spatial_w * w_stride;
          sh_h_low_ptr_offset[warp_id] = h_low * h_stride;
          sh_h_high_ptr_offset[warp_id] = h_high * h_stride;
          sh_w_low_ptr_offset[warp_id] = w_low * w_stride;
          sh_w_high_ptr_offset[warp_id] = w_high * w_stride;
          sh_do_compute[warp_id] = 1;
        } else {
          sh_do_compute[warp_id] = 0;
          sh_w1[warp_id] = sh_w2[warp_id] = sh_w3[warp_id] = sh_w4[warp_id] = 0.f;
          sh_valid1[warp_id] = sh_valid2[warp_id] = sh_valid3[warp_id] =
              sh_valid4[warp_id] = 0;
        }
      }
      __syncwarp();

      if (sh_do_compute[warp_id]) {
        msda_col2im_bilinear_grad_precomp_f32(
            data_value_ptr, spatial_h, spatial_w, channels,
            sh_h_low_ptr_offset[warp_id], sh_h_high_ptr_offset[warp_id],
            sh_w_low_ptr_offset[warp_id], sh_w_high_ptr_offset[warp_id],
            sh_lh[warp_id], sh_lw[warp_id], sh_hh[warp_id], sh_hw[warp_id],
            sh_w1[warp_id], sh_w2[warp_id], sh_w3[warp_id], sh_w4[warp_id],
            sh_valid1[warp_id], sh_valid2[warp_id], sh_valid3[warp_id],
            sh_valid4[warp_id], m_col, c_col, top_grad, weight, grad_value_ptr,
            &gw, &gh, &ga);
      }

      msda_warp_reduce_sum3_f32(gw, gh, ga);
      if (lane == 0) {
        grad_sampling_ptr_base[0] = gw;
        grad_sampling_ptr_base[1] = gh;
        *grad_attn_ptr_base = ga;
      }

      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
      grad_attn_ptr_base += 1;
      grad_sampling_ptr_base += 2;
    }
  }
  (void)batch_size;
}

// SpecA/SpecB bwd：对 (NUM_LEVELS, NUM_POINTS) 做编译期展开，减少循环与部分索引开销。
template <int NUM_LEVELS, int NUM_POINTS>
__global__ void ms_deform_col2im_channels32_warp_reduce_f32_kernel_spec(
    const int n, const float *grad_col, int64_t g_stride_b, int64_t g_stride_q,
    int64_t g_stride_m, int64_t g_stride_c, bool grad_col_contiguous,
    const float *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, const int batch_size, const int spatial_size,
    const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, float *grad_value,
    float *grad_sampling_loc, float *grad_attn_weight,
    at::musa::FastDivmod fastdivmod_channels, at::musa::FastDivmod fastdivmod_num_heads,
    at::musa::FastDivmod fastdivmod_num_query) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    return;

  uint32_t _temp = static_cast<uint32_t>(index);
  uint32_t c_col;
  uint32_t sampling_index;
  uint32_t m_col;
  uint32_t q_col;
  uint32_t b_col;
  fastdivmod_channels(sampling_index, c_col, _temp);
  fastdivmod_num_heads(_temp, m_col, sampling_index);
  fastdivmod_num_query(b_col, q_col, _temp);

  const float top_grad = msda_grad_col_load(
      grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c, b_col, q_col,
      m_col, c_col, grad_col_contiguous, channels, num_heads, num_query);

  int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINTS;
  int data_loc_w_ptr = data_weight_ptr << 1;
  const int grad_sampling_ptr = data_weight_ptr;
  float *grad_sampling_ptr_base = grad_sampling_loc + (grad_sampling_ptr << 1);
  float *grad_attn_ptr_base = grad_attn_weight + grad_sampling_ptr;

  const int qid_stride = num_heads * channels;
  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

  const int lane = threadIdx.x & 31;
  // Per-warp slot so multiple warps per block don't stomp the geometry cache.
  const int warp_id = threadIdx.x >> 5;

  // Geometry cache (same as non-spec kernel), per-warp arrays.
  __shared__ int sh_do_compute[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_valid1[MSDA_COL2IM_MAX_WARPS], sh_valid2[MSDA_COL2IM_MAX_WARPS],
      sh_valid3[MSDA_COL2IM_MAX_WARPS], sh_valid4[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_lh[MSDA_COL2IM_MAX_WARPS], sh_lw[MSDA_COL2IM_MAX_WARPS],
      sh_hh[MSDA_COL2IM_MAX_WARPS], sh_hw[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_w1[MSDA_COL2IM_MAX_WARPS], sh_w2[MSDA_COL2IM_MAX_WARPS],
      sh_w3[MSDA_COL2IM_MAX_WARPS], sh_w4[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_h_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_h_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_w_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_w_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];

  for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
    const int64_t level_start_id = data_level_start_index[l_col];
    const int spatial_h_ptr = l_col << 1;
    const int spatial_h =
        static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
    const int spatial_w =
        static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
    const int value_ptr_offset =
        data_value_ptr_init_offset + level_start_id * qid_stride;
    const float *data_value_ptr = data_value + value_ptr_offset;
    float *grad_value_ptr = grad_value + value_ptr_offset;

    for (int p_col = 0; p_col < NUM_POINTS; ++p_col) {
      const float loc_w = data_sampling_loc[data_loc_w_ptr];
      const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
      const float weight = data_attn_weight[data_weight_ptr];

      const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
      const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;

      float gw = 0.f, gh = 0.f, ga = 0.f;

      if (lane == 0) {
        const bool in_range =
            (h_im > -1.f && w_im > -1.f && h_im < static_cast<float>(spatial_h) &&
             w_im < static_cast<float>(spatial_w));
        if (in_range) {
          const int h_low = floorf(h_im);
          const int w_low = floorf(w_im);
          const int h_high = h_low + 1;
          const int w_high = w_low + 1;

          sh_lh[warp_id] = h_im - static_cast<float>(h_low);
          sh_lw[warp_id] = w_im - static_cast<float>(w_low);
          sh_hh[warp_id] = 1.f - sh_lh[warp_id];
          sh_hw[warp_id] = 1.f - sh_lw[warp_id];

          sh_w1[warp_id] = sh_hh[warp_id] * sh_hw[warp_id];
          sh_w2[warp_id] = sh_hh[warp_id] * sh_lw[warp_id];
          sh_w3[warp_id] = sh_lh[warp_id] * sh_hw[warp_id];
          sh_w4[warp_id] = sh_lh[warp_id] * sh_lw[warp_id];

          sh_valid1[warp_id] = (h_low >= 0 && w_low >= 0);
          sh_valid2[warp_id] = (h_low >= 0 && w_high <= spatial_w - 1);
          sh_valid3[warp_id] = (h_high <= spatial_h - 1 && w_low >= 0);
          sh_valid4[warp_id] =
              (h_high <= spatial_h - 1 && w_high <= spatial_w - 1);

          const int w_stride = num_heads * channels;
          const int h_stride = spatial_w * w_stride;
          sh_h_low_ptr_offset[warp_id] = h_low * h_stride;
          sh_h_high_ptr_offset[warp_id] = h_high * h_stride;
          sh_w_low_ptr_offset[warp_id] = w_low * w_stride;
          sh_w_high_ptr_offset[warp_id] = w_high * w_stride;
          sh_do_compute[warp_id] = 1;
        } else {
          sh_do_compute[warp_id] = 0;
          sh_w1[warp_id] = sh_w2[warp_id] = sh_w3[warp_id] = sh_w4[warp_id] = 0.f;
          sh_valid1[warp_id] = sh_valid2[warp_id] = sh_valid3[warp_id] =
              sh_valid4[warp_id] = 0;
        }
      }
      __syncwarp();

      if (sh_do_compute[warp_id]) {
        msda_col2im_bilinear_grad_precomp_f32(
            data_value_ptr, spatial_h, spatial_w, channels,
            sh_h_low_ptr_offset[warp_id], sh_h_high_ptr_offset[warp_id],
            sh_w_low_ptr_offset[warp_id], sh_w_high_ptr_offset[warp_id],
            sh_lh[warp_id], sh_lw[warp_id], sh_hh[warp_id], sh_hw[warp_id],
            sh_w1[warp_id], sh_w2[warp_id], sh_w3[warp_id], sh_w4[warp_id],
            sh_valid1[warp_id], sh_valid2[warp_id], sh_valid3[warp_id],
            sh_valid4[warp_id], m_col, c_col, top_grad, weight, grad_value_ptr,
            &gw, &gh, &ga);
      }

      msda_warp_reduce_sum3_f32(gw, gh, ga);
      if (lane == 0) {
        grad_sampling_ptr_base[0] = gw;
        grad_sampling_ptr_base[1] = gh;
        *grad_attn_ptr_base = ga;
      }

      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
      grad_attn_ptr_base += 1;
      grad_sampling_ptr_base += 2;
    }
  }

  (void)batch_size;
}

// ---------------------------------------------------------------------------
// fp16 (__half) col2im backward.
//
// CRITICAL numerical contract (per temporal_self_attention.py:242 — "fp16
// deformable attention is unstable because it performs many sum operations"):
//   * grad_value atomic accumulation is done in FLOAT. grad_value points to an
//     fp32 buffer; atomicAdd targets float*, so the 4-level x 8-point x 32-ch
//     scatter sums never round in fp16. dispatch.py allocates this fp32 buffer
//     and narrows to __half once at the end.
//   * grad_sampling_loc / grad_attn_weight warp-shuffle reduction is in FLOAT;
//     only the final per-(level,point) result is narrowed to __half on store.
//   * data_value / grad_col(top_grad) / loc / weight are loaded __half then
//     widened to float; all arithmetic is float.
//
// fp32 precomp grad helper, but reading __half value into float and scattering
// into an fp32 grad_value buffer.
__device__ inline void msda_col2im_bilinear_grad_precomp_f16(
    const __half *bottom_data, int height, int width, int channels,
    int h_low_ptr_offset, int h_high_ptr_offset, int w_low_ptr_offset,
    int w_high_ptr_offset, float lh, float lw, float hh, float hw, float w1,
    float w2, float w3, float w4, int valid1, int valid2, int valid3,
    int valid4, int m, int c, float top_grad, float attn_weight,
    float *grad_value, float *grad_w_acc, float *grad_h_acc,
    float *grad_a_acc) {
  const float top_grad_value = top_grad * attn_weight;
  float grad_h_weight = 0.f, grad_w_weight = 0.f;

  float v1 = 0.f;
  if (valid1) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + m * channels + c;
    v1 = __half2float(bottom_data[ptr1]);
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);  // fp32 atomic
  }

  float v2 = 0.f;
  if (valid2) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + m * channels + c;
    v2 = __half2float(bottom_data[ptr2]);
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }

  float v3 = 0.f;
  if (valid3) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + m * channels + c;
    v3 = __half2float(bottom_data[ptr3]);
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }

  float v4 = 0.f;
  if (valid4) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + m * channels + c;
    v4 = __half2float(bottom_data[ptr4]);
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_a_acc = top_grad * val;
  *grad_w_acc = static_cast<float>(width) * grad_w_weight * top_grad_value;
  *grad_h_acc = static_cast<float>(height) * grad_h_weight * top_grad_value;
}

// __half grad_col load (mirrors msda_grad_col_load but widens to float).
__device__ inline float msda_grad_col_load_f16(
    const __half *grad_col, int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int b, int q, int m, int c, bool contig, int channels, int num_heads,
    int num_query) {
  if (contig) {
    const int idx = ((b * num_query + q) * num_heads + m) * channels + c;
    return __half2float(grad_col[idx]);
  }
  return __half2float(grad_col[b * s0 + q * s1 + m * s2 + c * s3]);
}

template <int NUM_LEVELS, int NUM_POINTS>
__global__ void ms_deform_col2im_channels32_warp_reduce_f16_kernel_spec(
    const int n, const __half *grad_col, int64_t g_stride_b, int64_t g_stride_q,
    int64_t g_stride_m, int64_t g_stride_c, bool grad_col_contiguous,
    const __half *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const __half *data_sampling_loc,
    const __half *data_attn_weight, const int batch_size, const int spatial_size,
    const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, float *grad_value,
    __half *grad_sampling_loc, __half *grad_attn_weight,
    at::musa::FastDivmod fastdivmod_channels,
    at::musa::FastDivmod fastdivmod_num_heads,
    at::musa::FastDivmod fastdivmod_num_query) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n)
    return;

  uint32_t _temp = static_cast<uint32_t>(index);
  uint32_t c_col;
  uint32_t sampling_index;
  uint32_t m_col;
  uint32_t q_col;
  uint32_t b_col;
  fastdivmod_channels(sampling_index, c_col, _temp);
  fastdivmod_num_heads(_temp, m_col, sampling_index);
  fastdivmod_num_query(b_col, q_col, _temp);

  const float top_grad = msda_grad_col_load_f16(
      grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c, b_col, q_col,
      m_col, c_col, grad_col_contiguous, channels, num_heads, num_query);

  int data_weight_ptr = sampling_index * NUM_LEVELS * NUM_POINTS;
  int data_loc_w_ptr = data_weight_ptr << 1;
  const int grad_sampling_ptr = data_weight_ptr;
  __half *grad_sampling_ptr_base = grad_sampling_loc + (grad_sampling_ptr << 1);
  __half *grad_attn_ptr_base = grad_attn_weight + grad_sampling_ptr;

  const int qid_stride = num_heads * channels;
  const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  __shared__ int sh_do_compute[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_valid1[MSDA_COL2IM_MAX_WARPS], sh_valid2[MSDA_COL2IM_MAX_WARPS],
      sh_valid3[MSDA_COL2IM_MAX_WARPS], sh_valid4[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_lh[MSDA_COL2IM_MAX_WARPS], sh_lw[MSDA_COL2IM_MAX_WARPS],
      sh_hh[MSDA_COL2IM_MAX_WARPS], sh_hw[MSDA_COL2IM_MAX_WARPS];
  __shared__ float sh_w1[MSDA_COL2IM_MAX_WARPS], sh_w2[MSDA_COL2IM_MAX_WARPS],
      sh_w3[MSDA_COL2IM_MAX_WARPS], sh_w4[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_h_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_h_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];
  __shared__ int sh_w_low_ptr_offset[MSDA_COL2IM_MAX_WARPS],
      sh_w_high_ptr_offset[MSDA_COL2IM_MAX_WARPS];

  for (int l_col = 0; l_col < NUM_LEVELS; ++l_col) {
    const int64_t level_start_id = data_level_start_index[l_col];
    const int spatial_h_ptr = l_col << 1;
    const int spatial_h = static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
    const int spatial_w =
        static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
    const int value_ptr_offset =
        data_value_ptr_init_offset + level_start_id * qid_stride;
    const __half *data_value_ptr = data_value + value_ptr_offset;
    float *grad_value_ptr = grad_value + value_ptr_offset;  // fp32 buffer

    for (int p_col = 0; p_col < NUM_POINTS; ++p_col) {
      const float loc_w = __half2float(data_sampling_loc[data_loc_w_ptr]);
      const float loc_h = __half2float(data_sampling_loc[data_loc_w_ptr + 1]);
      const float weight = __half2float(data_attn_weight[data_weight_ptr]);

      const float h_im = loc_h * static_cast<float>(spatial_h) - 0.5f;
      const float w_im = loc_w * static_cast<float>(spatial_w) - 0.5f;

      float gw = 0.f, gh = 0.f, ga = 0.f;

      if (lane == 0) {
        const bool in_range =
            (h_im > -1.f && w_im > -1.f && h_im < static_cast<float>(spatial_h) &&
             w_im < static_cast<float>(spatial_w));
        if (in_range) {
          const int h_low = floorf(h_im);
          const int w_low = floorf(w_im);
          const int h_high = h_low + 1;
          const int w_high = w_low + 1;

          sh_lh[warp_id] = h_im - static_cast<float>(h_low);
          sh_lw[warp_id] = w_im - static_cast<float>(w_low);
          sh_hh[warp_id] = 1.f - sh_lh[warp_id];
          sh_hw[warp_id] = 1.f - sh_lw[warp_id];

          sh_w1[warp_id] = sh_hh[warp_id] * sh_hw[warp_id];
          sh_w2[warp_id] = sh_hh[warp_id] * sh_lw[warp_id];
          sh_w3[warp_id] = sh_lh[warp_id] * sh_hw[warp_id];
          sh_w4[warp_id] = sh_lh[warp_id] * sh_lw[warp_id];

          sh_valid1[warp_id] = (h_low >= 0 && w_low >= 0);
          sh_valid2[warp_id] = (h_low >= 0 && w_high <= spatial_w - 1);
          sh_valid3[warp_id] = (h_high <= spatial_h - 1 && w_low >= 0);
          sh_valid4[warp_id] =
              (h_high <= spatial_h - 1 && w_high <= spatial_w - 1);

          const int w_stride = num_heads * channels;
          const int h_stride = spatial_w * w_stride;
          sh_h_low_ptr_offset[warp_id] = h_low * h_stride;
          sh_h_high_ptr_offset[warp_id] = h_high * h_stride;
          sh_w_low_ptr_offset[warp_id] = w_low * w_stride;
          sh_w_high_ptr_offset[warp_id] = w_high * w_stride;
          sh_do_compute[warp_id] = 1;
        } else {
          sh_do_compute[warp_id] = 0;
          sh_w1[warp_id] = sh_w2[warp_id] = sh_w3[warp_id] = sh_w4[warp_id] = 0.f;
          sh_valid1[warp_id] = sh_valid2[warp_id] = sh_valid3[warp_id] =
              sh_valid4[warp_id] = 0;
        }
      }
      __syncwarp();

      if (sh_do_compute[warp_id]) {
        msda_col2im_bilinear_grad_precomp_f16(
            data_value_ptr, spatial_h, spatial_w, channels,
            sh_h_low_ptr_offset[warp_id], sh_h_high_ptr_offset[warp_id],
            sh_w_low_ptr_offset[warp_id], sh_w_high_ptr_offset[warp_id],
            sh_lh[warp_id], sh_lw[warp_id], sh_hh[warp_id], sh_hw[warp_id],
            sh_w1[warp_id], sh_w2[warp_id], sh_w3[warp_id], sh_w4[warp_id],
            sh_valid1[warp_id], sh_valid2[warp_id], sh_valid3[warp_id],
            sh_valid4[warp_id], m_col, c_col, top_grad, weight, grad_value_ptr,
            &gw, &gh, &ga);
      }

      msda_warp_reduce_sum3_f32(gw, gh, ga);  // float reduce
      if (lane == 0) {
        grad_sampling_ptr_base[0] = __float2half(gw);
        grad_sampling_ptr_base[1] = __float2half(gh);
        *grad_attn_ptr_base = __float2half(ga);
      }

      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
      grad_attn_ptr_base += 1;
      grad_sampling_ptr_base += 2;
    }
  }

  (void)batch_size;
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_shm_stride_v1_kernel(
    const int n, const scalar_t *grad_col, int64_t g_stride_b, int64_t g_stride_q,
    int64_t g_stride_m, int64_t g_stride_c, bool grad_col_contiguous,
    const scalar_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *data_sampling_loc,
    const scalar_t *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    scalar_t *grad_value, scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight,
    at::musa::FastDivmod fastdivmod_channels, at::musa::FastDivmod fastdivmod_num_heads, 
    at::musa::FastDivmod fastdivmod_num_query) {
  MSDA_1D_KERNEL_LOOP(index, n) {
    extern __shared__ int msda_shm[];
    scalar_t *cache_grad_sampling_loc =
        reinterpret_cast<scalar_t *>(msda_shm);
    scalar_t *cache_grad_attn_weight =
        cache_grad_sampling_loc + 2 * blockDim.x;
    const unsigned int tid = threadIdx.x;
    uint32_t _temp = index;
    uint32_t c_col;
    uint32_t sampling_index;
    uint32_t m_col;
    uint32_t q_col;
    uint32_t b_col;
    fastdivmod_channels(sampling_index, c_col, _temp);
    fastdivmod_num_heads(_temp, m_col, sampling_index);
    fastdivmod_num_query(b_col, q_col, _temp);

    const scalar_t top_grad = msda_grad_col_load(
        grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c, b_col, q_col,
        m_col, c_col, grad_col_contiguous, channels, num_heads, num_query);

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_base =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_base = grad_attn_weight + grad_sampling_ptr;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = static_cast<int>(data_spatial_shapes[spatial_h_ptr]);
      const int spatial_w =
          static_cast<int>(data_spatial_shapes[spatial_h_ptr + 1]);
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im =
            loc_h * static_cast<scalar_t>(spatial_h) -
            static_cast<scalar_t>(0.5);
        const scalar_t w_im =
            loc_w * static_cast<scalar_t>(spatial_w) -
            static_cast<scalar_t>(0.5);
        cache_grad_sampling_loc[threadIdx.x << 1] = 0;
        cache_grad_sampling_loc[(threadIdx.x << 1) + 1] = 0;
        cache_grad_attn_weight[threadIdx.x] = 0;
        if (h_im > static_cast<scalar_t>(-1) && w_im > static_cast<scalar_t>(-1) &&
            h_im < static_cast<scalar_t>(spatial_h) &&
            w_im < static_cast<scalar_t>(spatial_w)) {
          msda_col2im_bilinear_grad(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_sampling_loc + (threadIdx.x << 1) + 1,
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int j = 1; j < blockDim.x; ++j) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[j];
            sid += 2;
          }
          grad_sampling_base[0] = _grad_w;
          grad_sampling_base[1] = _grad_h;
          *grad_attn_base = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_base += 1;
        grad_sampling_base += 2;
      }
    }
  }
}
