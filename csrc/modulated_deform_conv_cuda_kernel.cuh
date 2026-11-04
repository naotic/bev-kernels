/*
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License
 * Modulated Deformable im2col / col2im kernels
 * Ref: https://arxiv.org/abs/1703.06211
 *
 * Extracted from mmcv for standalone CUDA build.
 */

#ifndef MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH
#define MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH

#include <float.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

using at::Half;
using at::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

template <typename T>
__device__ __forceinline__ T dmcn_im2col_bilinear(const T *input, const int data_width,
                                  const int height, const int width, T h, T w) {
  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__device__ T dmcn_get_gradient_weight(T argmax_h, T argmax_w, const int h,
                                      const int w, const int height,
                                      const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
__device__ T dmcn_get_coordinate_weight(T argmax_h, T argmax_w,
                                        const int height, const int width,
                                        const T *im_data, const int data_width,
                                        const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T>
__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n, const T *data_im, const T *data_offset, const T *data_mask,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;

    const T *data_mask_ptr =
        data_mask + (b_col * deformable_group + deformable_group_index) *
                        kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im,
                                     w_im);
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
__global__ void modulated_deformable_col2im_gpu_kernel(
    const int n, const T *data_col, const T *data_offset, const T *data_mask,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    T *grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight =
              dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data,
                                       cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename T>
__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n, const T *data_col, const T *data_im, const T *data_offset,
    const T *data_mask, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col, T *grad_offset, T *grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T *data_col_ptr = data_col + deformable_group_index *
                                           channel_per_deformable_group *
                                           batch_size * width_col * height_col;
    const T *data_im_ptr =
        data_im + (b * deformable_group + deformable_group_index) *
                      channel_per_deformable_group / kernel_h / kernel_w *
                      height * width;
    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      const T mask = data_mask_ptr[data_mask_hw_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
        inv_h = inv_w = -2;
      else
        mval += data_col_ptr[col_pos] *
                dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width,
                                     height, width, inv_h, inv_w);
      const T weight = dmcn_get_coordinate_weight(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
  }
}

// =====================================================================
// =====================================================================
// Optimized im2col: 3D grid with channel tiling for adaptive parallelism.
//
// Grid:  (spatial_blocks, batch_size, kernel_h * chan_chunks)
//   blockIdx.x * blockDim.x + threadIdx.x  → spatial index (h_col, w_col)
//   blockIdx.y                              → batch index b
//   blockIdx.z                              → combined (ki, chunk_id)
//
// Channel tiling: C channels are split into chan_chunks of size CTILE each.
// Each thread handles ONE spatial position, ONE kernel row, ONE channel chunk.
// Offset/mask are read once per (ki, kj, dg) and reused across CTILE channels.
//
// When chan_chunks=1 this is identical to the previous non-tiled version.
// When parallelism is low (small batch/spatial), more chunks = more blocks.
// =====================================================================

template <typename T>
__global__ void modulated_deformable_im2col_opt_gpu_kernel(
    const T* __restrict__ data_im,
    const T* __restrict__ data_offset,
    const T* __restrict__ data_mask,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size, const int num_channels,
    const int deformable_group,
    const int height_col, const int width_col,
    const int chan_chunks,
    T* __restrict__ data_col) {

    const int hw_col = height_col * width_col;
    const int bhw = batch_size * hw_col;
    const int kernel_size = kernel_h * kernel_w;
    const int im_hw = height * width;
    const int cpg = num_channels / deformable_group;
    const int ctile = cpg / chan_chunks;

    const int sp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp_idx >= hw_col) return;

    const int b = blockIdx.y;
    const int z = blockIdx.z;
    const int ki = z / chan_chunks;
    const int chunk_id = z - ki * chan_chunks;

    const int h_col = sp_idx / width_col;
    const int w_col = sp_idx - h_col * width_col;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    const T* im_base = data_im + b * num_channels * im_hw;
    const int sp_out = b * hw_col + sp_idx;
    const int sp_off = h_col * width_col + w_col;

    const float h_base = h_in + ki * dilation_h;

    for (int kj = 0; kj < kernel_w; kj++) {
        const int k = ki * kernel_w + kj;
        const float w_base = w_in + kj * dilation_w;

        for (int dg = 0; dg < deformable_group; dg++) {
            const int dg_off_base =
                (b * deformable_group + dg) * 2 * kernel_size * hw_col;
            const int off_idx = dg_off_base + (2 * k) * hw_col + sp_off;
            const float offset_h = static_cast<float>(data_offset[off_idx]);
            const float offset_w = static_cast<float>(data_offset[off_idx + hw_col]);

            const int dg_mask_base =
                (b * deformable_group + dg) * kernel_size * hw_col;
            const T cur_mask = data_mask[dg_mask_base + k * hw_col + sp_off];

            const float h_im = h_base + offset_h;
            const float w_im = w_base + offset_w;

            const bool in_bounds =
                h_im > -1.0f && w_im > -1.0f &&
                h_im < static_cast<float>(height) &&
                w_im < static_cast<float>(width);

            const int h_low = floorf(h_im);
            const int w_low = floorf(w_im);
            const int h_high = h_low + 1;
            const int w_high = w_low + 1;

            const float lh = h_im - h_low;
            const float lw = w_im - w_low;
            const float hh = 1.0f - lh;
            const float hw = 1.0f - lw;
            const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

            const bool b_hl_wl = (h_low >= 0 && w_low >= 0);
            const bool b_hl_wh = (h_low >= 0 && w_high <= width - 1);
            const bool b_hh_wl = (h_high <= height - 1 && w_low >= 0);
            const bool b_hh_wh = (h_high <= height - 1 && w_high <= width - 1);

            const int p1 = h_low  * width + w_low;
            const int p2 = h_low  * width + w_high;
            const int p3 = h_high * width + w_low;
            const int p4 = h_high * width + w_high;

            const int c_start = dg * cpg + chunk_id * ctile;
            const int c_end = c_start + ctile;
            const T* im_ptr = im_base + c_start * im_hw;
            int col_idx = (c_start * kernel_size + k) * bhw + sp_out;
            const int col_step = kernel_size * bhw;

            const int tile_pairs = ctile >> 1;
            const int tile_tail  = ctile & 1;

            for (int ci = 0; ci < tile_pairs; ci++) {
                T val0 = static_cast<T>(0);
                T val1 = static_cast<T>(0);
                if (in_bounds) {
                    float f0 = w1 * static_cast<float>(b_hl_wl ? im_ptr[p1] : static_cast<T>(0))
                             + w2 * static_cast<float>(b_hl_wh ? im_ptr[p2] : static_cast<T>(0))
                             + w3 * static_cast<float>(b_hh_wl ? im_ptr[p3] : static_cast<T>(0))
                             + w4 * static_cast<float>(b_hh_wh ? im_ptr[p4] : static_cast<T>(0));
                    val0 = static_cast<T>(f0);
                    const T* im_next = im_ptr + im_hw;
                    float f1 = w1 * static_cast<float>(b_hl_wl ? im_next[p1] : static_cast<T>(0))
                             + w2 * static_cast<float>(b_hl_wh ? im_next[p2] : static_cast<T>(0))
                             + w3 * static_cast<float>(b_hh_wl ? im_next[p3] : static_cast<T>(0))
                             + w4 * static_cast<float>(b_hh_wh ? im_next[p4] : static_cast<T>(0));
                    val1 = static_cast<T>(f1);
                }
                data_col[col_idx] = val0 * cur_mask;
                data_col[col_idx + col_step] = val1 * cur_mask;
                im_ptr  += im_hw + im_hw;
                col_idx += col_step + col_step;
            }
            if (tile_tail) {
                T val = static_cast<T>(0);
                if (in_bounds) {
                    float f = w1 * static_cast<float>(b_hl_wl ? im_ptr[p1] : static_cast<T>(0))
                            + w2 * static_cast<float>(b_hl_wh ? im_ptr[p2] : static_cast<T>(0))
                            + w3 * static_cast<float>(b_hh_wl ? im_ptr[p3] : static_cast<T>(0))
                            + w4 * static_cast<float>(b_hh_wh ? im_ptr[p4] : static_cast<T>(0));
                    val = static_cast<T>(f);
                }
                data_col[col_idx] = val * cur_mask;
            }
        }
    }
}

// =====================================================================
// Optimized col2im: 3D grid + direct 4-point scatter (no 5x5 scan)
//
// Grid:  (spatial_blocks, batch_size, kernel_h * chan_chunks)
//   blockIdx.x * blockDim.x + threadIdx.x  → spatial index (h_col, w_col)
//   blockIdx.y                              → batch index b
//   blockIdx.z                              → combined (ki, chunk_id)
//
// Key optimizations vs baseline:
//   1. Eliminates 5x5 scan loop — directly computes 4 bilinear corner points.
//   2. Replaces integer division chain with 3D grid → ~6 int divs eliminated.
//   3. Inlines gradient weight as hh*hw / hh*lw / lh*hw / lh*lw — no branches.
//   4. Channel tiling: multiple channels per thread reuse offset/mask reads.
//   5. __restrict__ for pointer non-aliasing optimization.
//
// Note: __launch_bounds__ is omitted because mcc 4.3 (mp_31) ignores it.
//       Use --mtgpu-maxregcnt via setup.py extra_compile_args if register
//       pressure tuning is needed.
// =====================================================================

template <typename T>
__global__ void
modulated_deformable_col2im_opt_gpu_kernel(
    const T* __restrict__ data_col,
    const T* __restrict__ data_offset,
    const T* __restrict__ data_mask,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    const int chan_chunks,
    T* __restrict__ grad_im) {

    const int hw_col = height_col * width_col;
    const int bhw = batch_size * hw_col;
    const int kernel_size = kernel_h * kernel_w;
    const int im_hw = height * width;
    const int cpg = channels / deformable_group;
    const int ctile = cpg / chan_chunks;

    const int sp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp_idx >= hw_col) return;

    const int b = blockIdx.y;
    const int z = blockIdx.z;
    const int ki = z / chan_chunks;
    const int chunk_id = z - ki * chan_chunks;

    const int h_col = sp_idx / width_col;
    const int w_col = sp_idx - h_col * width_col;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    const int sp_col = b * hw_col + sp_idx;
    const int sp_off = h_col * width_col + w_col;

    const float h_base = h_in + ki * dilation_h;

    for (int kj = 0; kj < kernel_w; kj++) {
        const int k = ki * kernel_w + kj;
        const float w_base = w_in + kj * dilation_w;

        for (int dg = 0; dg < deformable_group; dg++) {
            const int dg_off_base =
                (b * deformable_group + dg) * 2 * kernel_size * hw_col;
            const int off_idx = dg_off_base + (2 * k) * hw_col + sp_off;
            const float offset_h = static_cast<float>(data_offset[off_idx]);
            const float offset_w = static_cast<float>(data_offset[off_idx + hw_col]);

            const int dg_mask_base =
                (b * deformable_group + dg) * kernel_size * hw_col;
            const float cur_mask = static_cast<float>(data_mask[dg_mask_base + k * hw_col + sp_off]);

            const float h_im = h_base + offset_h;
            const float w_im = w_base + offset_w;

            const bool in_bounds =
                h_im > -1.0f && w_im > -1.0f &&
                h_im < static_cast<float>(height) &&
                w_im < static_cast<float>(width);

            if (!in_bounds) continue;

            const int h_low = floorf(h_im);
            const int w_low = floorf(w_im);
            const int h_high = h_low + 1;
            const int w_high = w_low + 1;

            const float lh = h_im - h_low;
            const float lw = w_im - w_low;
            const float hh = 1.0f - lh;
            const float hw = 1.0f - lw;

            const float w1 = hh * hw;
            const float w2 = hh * lw;
            const float w3 = lh * hw;
            const float w4 = lh * lw;

            const bool b_hl = (h_low >= 0);
            const bool b_hh = (h_high <= height - 1);
            const bool b_wl = (w_low >= 0);
            const bool b_wh = (w_high <= width - 1);

            const int p1 = h_low  * width + w_low;
            const int p2 = h_low  * width + w_high;
            const int p3 = h_high * width + w_low;
            const int p4 = h_high * width + w_high;

            const int c_start = dg * cpg + chunk_id * ctile;
            const int c_end   = c_start + ctile;
            int col_idx = (c_start * kernel_size + k) * bhw + sp_col;
            const int col_step = kernel_size * bhw;

            T* grad_base = grad_im + (b * channels) * im_hw;

            for (int c = c_start; c < c_end; c++) {
                const float top_grad = static_cast<float>(data_col[col_idx]) * cur_mask;
                T* gptr = grad_base + c * im_hw;

                if (b_hl && b_wl) atomicAdd(gptr + p1, static_cast<T>(w1 * top_grad));
                if (b_hl && b_wh) atomicAdd(gptr + p2, static_cast<T>(w2 * top_grad));
                if (b_hh && b_wl) atomicAdd(gptr + p3, static_cast<T>(w3 * top_grad));
                if (b_hh && b_wh) atomicAdd(gptr + p4, static_cast<T>(w4 * top_grad));

                col_idx += col_step;
            }
        }
    }
}

// =====================================================================
// Shared-memory tiled col2im (v2)
//
// Each block owns an output tile of TILE_H × TILE_W spatial positions
// and ONE channel. It accumulates all kH×kW gradient contributions
// into a shared-memory buffer covering the corresponding input region,
// then writes the buffer back to global memory with a single atomicAdd
// per pixel (to handle overlap between adjacent tiles).
//
// Grid:  (tile_blocks_x, tile_blocks_y, batch_size * channels)
//   blockIdx.x  → tile column index
//   blockIdx.y  → tile row index
//   blockIdx.z  → (b * channels + c)
//
// Shared memory layout:
//   float smem[ SMEM_H * SMEM_W ]
//   where SMEM_H = TILE_H + 2*MARGIN, SMEM_W = TILE_W + 2*MARGIN
//   MARGIN accounts for kernel extent + maximum expected offset.
//
// Two phases:
//   Phase 1 (scatter): each thread processes one (h_out, w_out) in the
//     tile, loops over kH×kW, and atomicAdd's into smem.
//   Phase 2 (writeback): threads cooperatively write smem to grad_im.
//     Tiles may overlap in input space, so writeback uses atomicAdd
//     on global — but the count is O(SMEM_H×SMEM_W) per block, much
//     less than the O(TILE_H×TILE_W×kH×kW×4) global atomicAdds of v1.
// =====================================================================

// Tuning parameters:
//   TILE: 8×16 to better match typical W_out (100) alignment and increase
//         per-block work, reducing grid z pressure.
//   MARGIN=4: tighter than 5 — covers kernel extent (pad+2*dilation=3) plus
//         typical offset range (±1). The rare out-of-range fallback handles the rest.
//   SMEM_W_PAD = SMEM_W+1: eliminates bank conflicts (gcd(SMEM_W+1, 32)=1).
//   Block=128: 2× more threads to overlap smem atomicAdd latency with computation.

#define COL2IM_TILE_H 8
#define COL2IM_TILE_W 16
#define COL2IM_MARGIN 4

template <typename T>
__global__ void
modulated_deformable_col2im_smem_gpu_kernel(
    const T* __restrict__ data_col,
    const T* __restrict__ data_offset,
    const T* __restrict__ data_mask,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    T* __restrict__ grad_im) {

    const int SMEM_H = COL2IM_TILE_H + 2 * COL2IM_MARGIN;       // 16
    const int SMEM_W = COL2IM_TILE_W + 2 * COL2IM_MARGIN;       // 24
    const int SMEM_W_PAD = SMEM_W + 1;                           // 25 (bank-conflict-free)
    const int SMEM_SIZE = SMEM_H * SMEM_W_PAD;                   // 16×25 = 400

    extern __shared__ float smem[];

    const int hw_col = height_col * width_col;
    const int kernel_size = kernel_h * kernel_w;
    const int im_hw = height * width;
    const int cpg = channels / deformable_group;

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int bc = blockIdx.z;
    const int b = bc / channels;
    const int c = bc - b * channels;
    const int dg = c / cpg;

    const int tile_h_start = tile_row * COL2IM_TILE_H;
    const int tile_w_start = tile_col * COL2IM_TILE_W;

    const int in_h_origin = tile_h_start * stride_h - pad_h - COL2IM_MARGIN;
    const int in_w_origin = tile_w_start * stride_w - pad_w - COL2IM_MARGIN;

    // Phase 0: zero shared memory
    for (int i = threadIdx.x; i < SMEM_SIZE; i += blockDim.x)
        smem[i] = 0.0f;
    __syncthreads();

    // Phase 1: scatter
    const int tile_area = COL2IM_TILE_H * COL2IM_TILE_W;
    for (int tid = threadIdx.x; tid < tile_area; tid += blockDim.x) {
        const int local_h = tid / COL2IM_TILE_W;
        const int local_w = tid - local_h * COL2IM_TILE_W;
        const int h_out = tile_h_start + local_h;
        const int w_out = tile_w_start + local_w;

        if (h_out >= height_col || w_out >= width_col) continue;

        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;
        const int sp_off = h_out * width_col + w_out;
        const int sp_col = b * hw_col + sp_off;

        const int dg_off_base =
            (b * deformable_group + dg) * 2 * kernel_size * hw_col;
        const int dg_mask_base =
            (b * deformable_group + dg) * kernel_size * hw_col;

        for (int ki = 0; ki < kernel_h; ki++) {
            const float h_base = h_in + ki * dilation_h;
            for (int kj = 0; kj < kernel_w; kj++) {
                const int k = ki * kernel_w + kj;
                const float w_base = w_in + kj * dilation_w;

                const int off_idx = dg_off_base + (2 * k) * hw_col + sp_off;
                const float offset_h = static_cast<float>(data_offset[off_idx]);
                const float offset_w = static_cast<float>(data_offset[off_idx + hw_col]);
                const float cur_mask = static_cast<float>(
                    data_mask[dg_mask_base + k * hw_col + sp_off]);

                const float h_im = h_base + offset_h;
                const float w_im = w_base + offset_w;

                if (h_im <= -1.0f || w_im <= -1.0f ||
                    h_im >= static_cast<float>(height) ||
                    w_im >= static_cast<float>(width))
                    continue;

                const int col_idx =
                    (c * kernel_size + k) * (batch_size * hw_col) + sp_col;
                const float top_grad =
                    static_cast<float>(data_col[col_idx]) * cur_mask;

                const int h_low = floorf(h_im);
                const int w_low = floorf(w_im);
                const float lh = h_im - h_low;
                const float lw = w_im - w_low;
                const float hh = 1.0f - lh;
                const float hw = 1.0f - lw;

                const int sh = h_low - in_h_origin;
                const int sw = w_low - in_w_origin;

                if (sh >= 0 && sh + 1 < SMEM_H &&
                    sw >= 0 && sw + 1 < SMEM_W) {
                    float* sp = smem + sh * SMEM_W_PAD + sw;
                    if (h_low >= 0 && w_low >= 0)
                        atomicAdd(sp,                   hh * hw * top_grad);
                    if (h_low >= 0 && w_low + 1 <= width - 1)
                        atomicAdd(sp + 1,               hh * lw * top_grad);
                    if (h_low + 1 <= height - 1 && w_low >= 0)
                        atomicAdd(sp + SMEM_W_PAD,      lh * hw * top_grad);
                    if (h_low + 1 <= height - 1 && w_low + 1 <= width - 1)
                        atomicAdd(sp + SMEM_W_PAD + 1,  lh * lw * top_grad);
                } else {
                    T* gptr = grad_im + (b * channels + c) * im_hw;
                    if (h_low >= 0 && w_low >= 0)
                        atomicAdd(gptr + h_low * width + w_low,
                                  static_cast<T>(hh * hw * top_grad));
                    if (h_low >= 0 && w_low + 1 <= width - 1)
                        atomicAdd(gptr + h_low * width + w_low + 1,
                                  static_cast<T>(hh * lw * top_grad));
                    if (h_low + 1 <= height - 1 && w_low >= 0)
                        atomicAdd(gptr + (h_low + 1) * width + w_low,
                                  static_cast<T>(lh * hw * top_grad));
                    if (h_low + 1 <= height - 1 && w_low + 1 <= width - 1)
                        atomicAdd(gptr + (h_low + 1) * width + w_low + 1,
                                  static_cast<T>(lh * lw * top_grad));
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: writeback — iterate over logical (SMEM_H × SMEM_W) region,
    // reading from padded layout.
    const int wb_size = SMEM_H * SMEM_W;
    T* gptr = grad_im + (b * channels + c) * im_hw;
    for (int i = threadIdx.x; i < wb_size; i += blockDim.x) {
        const int sh = i / SMEM_W;
        const int sw = i - sh * SMEM_W;
        const float val = smem[sh * SMEM_W_PAD + sw];
        if (val == 0.0f) continue;
        const int gh = in_h_origin + sh;
        const int gw = in_w_origin + sw;
        if (gh >= 0 && gh < height && gw >= 0 && gw < width)
            atomicAdd(gptr + gh * width + gw, static_cast<T>(val));
    }
}

// =====================================================================
// col2im_coord_opt: optimized grad_offset / grad_mask computation.
//
// Optimizations vs baseline:
//   1. Loop-invariant hoisting: i, j, h_in, w_in, offset, mask, and all
//      address calculations that depend only on (offset_c, h, w, b) are
//      computed once before the channel loop.
//   2. Inlined bilinear + coordinate weight: the 4 corner pixel values
//      are loaded once per channel iteration and reused to compute both
//      the bilinear sample (for grad_mask) and the coordinate derivative
//      (for grad_offset), eliminating 2 device-function calls and their
//      redundant memory loads + branches.
//   3. __restrict__ pointers for non-aliasing optimization.
// =====================================================================

template <typename T>
__global__ void modulated_deformable_col2im_coord_opt_gpu_kernel(
    const int n,
    const T* __restrict__ data_col,
    const T* __restrict__ data_im,
    const T* __restrict__ data_offset,
    const T* __restrict__ data_mask,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col,
    T* __restrict__ grad_offset, T* __restrict__ grad_mask) {

  const int hw_col = height_col * width_col;
  const int im_hw = height * width;
  const int kernel_size = kernel_h * kernel_w;

  CUDA_1D_KERNEL_LOOP(index, n) {
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;

    const int dg = c / (2 * kernel_size);
    const int offset_c = c - dg * 2 * kernel_size;
    const int bp_dir = offset_c & 1;
    const int k = offset_c >> 1;
    const int ki = k / kernel_w;
    const int kj = k - ki * kernel_w;

    // --- Loop-invariant: offset, mask, sampling position (hoisted) ---
    const int sp_off = h * width_col + w;
    const int dg_off_base = (b * deformable_group + dg) * 2 * kernel_size * hw_col;
    const int off_idx = dg_off_base + (2 * k) * hw_col + sp_off;
    const float offset_h = static_cast<float>(data_offset[off_idx]);
    const float offset_w = static_cast<float>(data_offset[off_idx + hw_col]);

    const int dg_mask_base = (b * deformable_group + dg) * kernel_size * hw_col;
    const float mask = static_cast<float>(data_mask[dg_mask_base + k * hw_col + sp_off]);

    const int h_in = h * stride_h - pad_h;
    const int w_in = w * stride_w - pad_w;
    const float inv_h = h_in + ki * dilation_h + offset_h;
    const float inv_w = w_in + kj * dilation_w + offset_w;

    const bool in_bounds = inv_h > -1.0f && inv_w > -1.0f &&
                           inv_h < static_cast<float>(height) &&
                           inv_w < static_cast<float>(width);

    // --- Precompute bilinear corner weights (used by both grad paths) ---
    const int h_low = floorf(inv_h);
    const int w_low = floorf(inv_w);
    const float lh = inv_h - h_low;
    const float lw = inv_w - w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    const bool b_hl_wl = (h_low >= 0 && w_low >= 0);
    const bool b_hl_wh = (h_low >= 0 && w_low + 1 <= width - 1);
    const bool b_hh_wl = (h_low + 1 <= height - 1 && w_low >= 0);
    const bool b_hh_wh = (h_low + 1 <= height - 1 && w_low + 1 <= width - 1);

    const int p1 = h_low * width + w_low;
    const int p2 = p1 + 1;
    const int p3 = p1 + width;
    const int p4 = p3 + 1;

    // --- data_col / data_im base pointers ---
    const T* data_col_ptr = data_col + dg * channel_per_deformable_group *
                                       batch_size * hw_col;
    const int cpg_no_k = channel_per_deformable_group / kernel_size;
    const T* data_im_ptr = data_im + (b * deformable_group + dg) *
                                     cpg_no_k * im_hw;

    const int col_base = ((k * batch_size + b) * height_col + h) * width_col + w;
    const int col_stride = kernel_size * batch_size * hw_col;

    float val = 0.0f, mval = 0.0f;

    for (int cc = 0; cc < cpg_no_k; cc++) {
      const T* im_ptr = data_im_ptr + cc * im_hw;
      const float col_val = static_cast<float>(data_col_ptr[col_base + cc * col_stride]);

      if (in_bounds) {
        const float v1 = b_hl_wl ? static_cast<float>(im_ptr[p1]) : 0.0f;
        const float v2 = b_hl_wh ? static_cast<float>(im_ptr[p2]) : 0.0f;
        const float v3 = b_hh_wl ? static_cast<float>(im_ptr[p3]) : 0.0f;
        const float v4 = b_hh_wh ? static_cast<float>(im_ptr[p4]) : 0.0f;

        mval += col_val * (hh * hw * v1 + hh * lw * v2 +
                           lh * hw * v3 + lh * lw * v4);

        float coord_weight;
        if (bp_dir == 0) {
          coord_weight = -hw * v1 - lw * v2 + hw * v3 + lw * v4;
        } else {
          coord_weight = -hh * v1 + hh * v2 - lh * v3 + lh * v4;
        }
        val += coord_weight * col_val * mask;
      } else {
        // inv out of bounds: coord_weight = 0 (no contribution to val)
        // mval gets no contribution either
      }
    }

    grad_offset[index] = static_cast<T>(val);
    if (bp_dir == 0) {
      grad_mask[(((b * deformable_group + dg) * kernel_size + k) *
                     height_col + h) * width_col + w] = static_cast<T>(mval);
    }
  }
}

#endif  // MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH
