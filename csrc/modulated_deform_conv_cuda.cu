#include "modulated_deform_conv_cuda_kernel.cuh"

// forward im2col kernel
void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES(
      data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        modulated_deformable_im2col_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, data_offset_, data_mask_, height_im,
            width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, batch_size,
            channels, deformable_group, height_col, width_col, data_col_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "modulated_deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        modulated_deformable_col2im_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_offset_, data_mask_, channels,
            height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, deformable_group, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Optimized im2col: 3D grid with adaptive channel tiling.
void modulated_deformable_im2col_opt_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {

  const int hw_col = height_col * width_col;
  const int threads = 256;
  const int spatial_blocks = (hw_col + threads - 1) / threads;
  const int cpg = channels / deformable_group;

  const int base_blocks = spatial_blocks * batch_size * kernel_h;
  int num_sm = 1;
  {
    int device_id = data_im.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    num_sm = prop.multiProcessorCount;
  }
  const int target_blocks = num_sm * 4;
  const int min_ctile = 32;
  const int max_chunks = cpg / min_ctile;

  int chan_chunks = 1;
  if (base_blocks < target_blocks && max_chunks > 1) {
      for (int cc = 2; cc <= max_chunks; cc *= 2) {
          if (cpg % cc != 0) continue;
          chan_chunks = cc;
          if (base_blocks * cc >= target_blocks) break;
      }
  }

  dim3 grid(spatial_blocks, batch_size, kernel_h * chan_chunks);
  dim3 block(threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      data_im.scalar_type(), "modulated_deformable_im2col_opt_gpu", ([&] {
        modulated_deformable_im2col_opt_gpu_kernel<<<
            grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            data_im.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            data_mask.data_ptr<scalar_t>(),
            height_im, width_im, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w,
            batch_size, channels, deformable_group,
            height_col, width_col,
            chan_chunks,
            data_col.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Optimized col2im: 3D grid with direct 4-point scatter and adaptive channel tiling.
void modulated_deformable_col2im_opt_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {

  const int hw_col = height_col * width_col;
  const int threads = 256;
  const int spatial_blocks = (hw_col + threads - 1) / threads;
  const int cpg = channels / deformable_group;

  const int base_blocks = spatial_blocks * batch_size * kernel_h;
  int num_sm = 1;
  {
    int device_id = data_col.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    num_sm = prop.multiProcessorCount;
  }
  const int target_blocks = num_sm * 4;
  const int min_ctile = 32;
  const int max_chunks = cpg / min_ctile;

  int chan_chunks = 1;
  if (base_blocks < target_blocks && max_chunks > 1) {
      for (int cc = 2; cc <= max_chunks; cc *= 2) {
          if (cpg % cc != 0) continue;
          chan_chunks = cc;
          if (base_blocks * cc >= target_blocks) break;
      }
  }

  dim3 grid(spatial_blocks, batch_size, kernel_h * chan_chunks);
  dim3 block(threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      data_col.scalar_type(), "modulated_deformable_col2im_opt_gpu", ([&] {
        modulated_deformable_col2im_opt_gpu_kernel<<<
            grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            data_col.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            data_mask.data_ptr<scalar_t>(),
            channels, height_im, width_im, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w,
            batch_size, deformable_group,
            height_col, width_col,
            chan_chunks,
            grad_im.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Shared-memory tiled col2im: one block per (b, c, spatial_tile).
void modulated_deformable_col2im_smem_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {

  const int tile_cols = (width_col + COL2IM_TILE_W - 1) / COL2IM_TILE_W;
  const int tile_rows = (height_col + COL2IM_TILE_H - 1) / COL2IM_TILE_H;

  const int SMEM_H = COL2IM_TILE_H + 2 * COL2IM_MARGIN;
  const int SMEM_W = COL2IM_TILE_W + 2 * COL2IM_MARGIN;
  const int SMEM_W_PAD = SMEM_W + 1;
  const int smem_bytes = SMEM_H * SMEM_W_PAD * sizeof(float);

  const int threads = 128;
  dim3 grid(tile_cols, tile_rows, batch_size * channels);
  dim3 block(threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      data_col.scalar_type(), "modulated_deformable_col2im_smem_gpu", ([&] {
        modulated_deformable_col2im_smem_gpu_kernel<<<
            grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            data_col.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            data_mask.data_ptr<scalar_t>(),
            channels, height_im, width_im, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w,
            batch_size, deformable_group,
            height_col, width_col,
            grad_im.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                          kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_,
            channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size,
            2 * kernel_h * kernel_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_, grad_mask_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// col2im_coord_opt: baseline copy for incremental optimization.
void modulated_deformable_col2im_coord_opt_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                          kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_opt_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_opt_gpu_kernel<<<
            GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_,
            channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size,
            2 * kernel_h * kernel_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_, grad_mask_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
