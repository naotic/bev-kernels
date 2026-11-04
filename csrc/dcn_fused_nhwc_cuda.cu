#include <cstdlib>
#include <type_traits>

#include "dcn_fused_nhwc_kernel.cuh"

using at::Tensor;

// EXPERIMENT KNOB: dummy dynamic shared memory (bytes) to throttle occupancy on
// MUSA (mcc ignores __launch_bounds__/maxregcnt; smem is a hard occupancy cap:
// blocks/SM = 192KB / dummy). Set DCN_DUMMY_SMEM=<bytes> to reduce concurrent
// warps and test whether col2im is atomic-CONTENTION-bound (fewer issuers ->
// faster) vs throughput-bound (no change). 0 = full occupancy.
static int dcn_dummy_smem() {
  static const int v = []() {
    const char *e = std::getenv("DCN_DUMMY_SMEM");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

// Deformable gather (NHWC, dg==1):
//   data_im   [B, H, W, Cin]
//   data_offset [B, Ho, Wo, 2T]
//   data_mask   [B, Ho, Wo, T]
//   data_col  [M, T*Cin]  (M = B*Ho*Wo, k = t*Cin + ci)  -- pre-allocated.
// One warp per (m, t); lanes cover channels. See header for the rationale.
void modulated_deformable_im2col_nhwc_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  TORCH_CHECK(deformable_group == 1,
              "NHWC fused path supports deformable_group == 1 only");

  const long M = (long)batch_size * height_col * width_col;
  // kernel uses 32-bit indexing; largest index is the Col element count.
  TORCH_CHECK(M * kernel_h * kernel_w * channels < (long)2147483647,
              "NHWC gather: Col element count exceeds int32 range");

  const int Ttaps = kernel_h * kernel_w;
  const int threads = 256;                                  // 8 warps / block
  const int warps_per_block = threads / dcn_nhwc::WARP_SIZE;
  // Grid: x = ow tiles, y = oh, z = b*Ttaps + t. Coordinates read directly from
  // blockIdx -> no runtime division by Ho/Wo/HoWo in the kernel.
  dim3 grid((width_col + warps_per_block - 1) / warps_per_block, height_col,
            batch_size * Ttaps);
  dim3 block(threads);

  // FUSED_Y: data_offset is the raw conv output y [B,Ho,Wo,3T] (offset+raw mask);
  // detected by its last dim (3T vs 2T). When fused, offset/mask read from y + the
  // mask sigmoid happens inside the kernel.
  const bool fused_y = (data_offset.size(-1) == 3 * Ttaps);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, data_im.scalar_type(),
      "modulated_deformable_im2col_nhwc", ([&] {
        auto launch = [&](auto KH, auto KW, auto VW, auto FU) {
          dcn_nhwc::modulated_deformable_im2col_nhwc_gpu_kernel<
              scalar_t, decltype(KH)::value, decltype(KW)::value,
              decltype(VW)::value, decltype(FU)::value>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  data_im.data_ptr<scalar_t>(),
                  data_offset.data_ptr<scalar_t>(),
                  data_mask.data_ptr<scalar_t>(), batch_size, channels,
                  height_im, width_im, height_col, width_col, kernel_h,
                  kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                  dilation_w, data_col.data_ptr<scalar_t>());
        };
        auto disp_vw = [&](auto KH, auto KW, auto FU) {
          if (channels % 4 == 0)
            launch(KH, KW, std::integral_constant<int, 4>{}, FU);
          else
            launch(KH, KW, std::integral_constant<int, 2>{}, FU);
        };
        auto disp_kw = [&](auto FU) {
          if (kernel_h == 3 && kernel_w == 3)
            disp_vw(std::integral_constant<int, 3>{}, std::integral_constant<int, 3>{}, FU);
          else
            disp_vw(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, FU);
        };
        if (fused_y) disp_kw(std::true_type{});
        else disp_kw(std::false_type{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Fused F1-F5 (NHWC, dg==1): offset conv + bilinear gather + sigmoid modulation
// in one kernel.  w_off [Noff, T, Cin] (Noff=3*kH*kW), b_off [Noff] (float32),
// data_col [M, T*Cin] pre-allocated.  Main GEMM stays external.
void modulated_deformable_fused_im2col_nhwc_cuda(
    const Tensor data_im, const Tensor w_off, const Tensor b_off,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  TORCH_CHECK(deformable_group == 1,
              "NHWC fused path supports deformable_group == 1 only");
  TORCH_CHECK(kernel_h == 3 && kernel_w == 3,
              "NHWC fused kernel is specialized for 3x3 (use the split path otherwise)");
  const long M = (long)batch_size * height_col * width_col;
  TORCH_CHECK(M * kernel_h * kernel_w * channels < (long)2147483647,
              "NHWC fused: Col element count exceeds int32 range");

  const int Ttaps = kernel_h * kernel_w;
  const int threads = 256;
  const int warps_per_block = threads / dcn_nhwc::WARP_SIZE;
  dim3 grid((width_col + warps_per_block - 1) / warps_per_block, height_col,
            batch_size);                                   // one warp per pixel
  dim3 block(threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, data_im.scalar_type(),
      "modulated_deformable_fused_im2col_nhwc", ([&] {
        dcn_nhwc::modulated_deformable_fused_im2col_nhwc_gpu_kernel<scalar_t, 3, 3>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                data_im.data_ptr<scalar_t>(), w_off.data_ptr<scalar_t>(),
                b_off.data_ptr<float>(), batch_size, channels, height_im,
                width_im, height_col, width_col, kernel_h, kernel_w, pad_h,
                pad_w, stride_h, stride_w, dilation_h, dilation_w,
                data_col.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Backward col2im (NHWC, dg==1): dCol [M,T*Cin], data_im/offset/mask ->
// gX [B,H,W,Cin] (PRE-ZEROED, atomic), gOff [B,Ho,Wo,2T], gMask [B,Ho,Wo,T]
// (overwritten). gX/gOff/gMask are float32 accumulators.
void modulated_deformable_col2im_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask) {
  TORCH_CHECK(deformable_group == 1,
              "NHWC fused path supports deformable_group == 1 only");
  const long M = (long)batch_size * height_col * width_col;
  TORCH_CHECK(M * kernel_h * kernel_w * channels < (long)2147483647,
              "NHWC col2im: Col element count exceeds int32 range");

  const int Ttaps = kernel_h * kernel_w;
  const int threads = 256;
  const int warps_per_block = threads / dcn_nhwc::WARP_SIZE;
  dim3 grid((width_col + warps_per_block - 1) / warps_per_block, height_col,
            batch_size * Ttaps);
  dim3 block(threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_col2im_nhwc", ([&] {
        auto launch = [&](auto VW) {
          dcn_nhwc::modulated_deformable_col2im_nhwc_gpu_kernel<
              scalar_t, 3, 3, decltype(VW)::value, true>
              <<<grid, block, dcn_dummy_smem(), at::cuda::getCurrentCUDAStream()>>>(
                  dcol.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(),
                  data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
                  batch_size, channels, height_im, width_im, height_col,
                  width_col, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                  stride_w, dilation_h, dilation_w, grad_im.data_ptr<float>(),
                  grad_offset.data_ptr<float>(), grad_mask.data_ptr<float>());
        };
        TORCH_CHECK(kernel_h == 3 && kernel_w == 3,
                    "NHWC col2im specialized for 3x3");
        if (channels % 4 == 0)
          launch(std::integral_constant<int, 4>{});
        else
          launch(std::integral_constant<int, 2>{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// grad_offset/grad_mask only (atomic-free warp-reduce). Pair with the smem grad_X
// kernel below. gX-typed arg is unused (kept for kernel signature; never written).
void modulated_deformable_col2im_coord_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_offset,
    Tensor grad_mask) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3,
              "NHWC col2im_coord: dg==1, 3x3 only");
  const int Ttaps = kernel_h * kernel_w;
  const int threads = 256;
  const int wpb = threads / dcn_nhwc::WARP_SIZE;
  dim3 grid((width_col + wpb - 1) / wpb, height_col, batch_size * Ttaps);
  dim3 block(threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_col2im_coord_nhwc", ([&] {
        auto launch = [&](auto VW) {
          dcn_nhwc::modulated_deformable_col2im_nhwc_gpu_kernel<
              scalar_t, 3, 3, decltype(VW)::value, false>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  dcol.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(),
                  data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
                  batch_size, channels, height_im, width_im, height_col,
                  width_col, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                  stride_w, dilation_h, dilation_w,
                  grad_offset.data_ptr<float>() /*unused gX*/,
                  grad_offset.data_ptr<float>(), grad_mask.data_ptr<float>());
        };
        if (channels % 4 == 0)
          launch(std::integral_constant<int, 4>{});
        else
          launch(std::integral_constant<int, 2>{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// Shared-memory-tiled grad_X (3x3 stride-1 dil-1, channels % 32 == 0). gX PRE-ZEROED.
void modulated_deformable_gradx_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3 &&
                  stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
                  dilation_w == 1 && channels % 32 == 0,
              "smem grad_X: dg==1, 3x3, stride1, dil1, channels%32==0");
  constexpr int CC = 32, TH = 8, TW = 8, MG = 4;
  constexpr int SH = TH + 2 * MG, SW = TW + 2 * MG;
  const int nchunk = channels / CC;
  const int threads = 256;
  dim3 grid((width_col + TW - 1) / TW, (height_col + TH - 1) / TH,
            batch_size * nchunk);
  dim3 block(threads);
  const int smem_bytes = SH * SW * CC * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_gradx_smem_nhwc", ([&] {
        dcn_nhwc::modulated_deformable_col2im_gradx_smem_nhwc_gpu_kernel<
            scalar_t, 3, 3, CC, TH, TW, MG>
            <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                dcol.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(),
                data_mask.data_ptr<scalar_t>(), batch_size, channels, height_im,
                width_im, height_col, width_col, pad_h, pad_w, nchunk,
                grad_im.data_ptr<float>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// FULLY-FUSED B4-B9 with smem-tiled grad_X in ONE kernel. grad_im/grad_offset/
// grad_mask all PRE-ZEROED (all atomic-accumulated). 3x3 s1 d1, channels%32==0.
void modulated_deformable_col2im_fused_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3 &&
                  stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
                  dilation_w == 1 && channels % 32 == 0,
              "fused smem col2im: dg==1, 3x3, stride1, dil1, channels%32==0");
  constexpr int CC = 32, TH = 8, TW = 8, MG = 4;
  constexpr int SH = TH + 2 * MG, SW = TW + 2 * MG;
  const int nchunk = channels / CC;
  const int threads = 256;
  dim3 grid((width_col + TW - 1) / TW, (height_col + TH - 1) / TH,
            batch_size * nchunk);
  dim3 block(threads);
  const int smem_bytes = SH * SW * CC * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_col2im_fused_smem_nhwc", ([&] {
        dcn_nhwc::modulated_deformable_col2im_fused_smem_nhwc_gpu_kernel<
            scalar_t, 3, 3, CC, TH, TW, MG>
            <<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
                dcol.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(),
                data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
                batch_size, channels, height_im, width_im, height_col, width_col,
                pad_h, pad_w, nchunk, grad_im.data_ptr<float>(),
                grad_offset.data_ptr<float>(), grad_mask.data_ptr<float>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// PER-ELEMENT col2im (TileLang-style, grid-decoded). One thread per (m,t,ci):
// block = channels, grid = (Wo, Ho, B*T). grad_im/offset/mask PRE-ZEROED.
void modulated_deformable_col2im_perelem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3 &&
                  stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
                  dilation_w == 1 && channels % 32 == 0 && channels <= 1024,
              "perelem col2im: dg==1, 3x3, stride1, dil1, channels%32==0, <=1024");
  dim3 grid(width_col, height_col, batch_size * kernel_h * kernel_w);
  dim3 block(channels);
  const bool fused_y = (data_offset.size(-1) == 3 * kernel_h * kernel_w);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_col2im_perelem_nhwc", ([&] {
        auto launch = [&](auto FU) {
          dcn_nhwc::modulated_deformable_col2im_perelem_nhwc_gpu_kernel<
              scalar_t, 3, 3, decltype(FU)::value>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  dcol.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(),
                  data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
                  batch_size, channels, height_im, width_im, height_col, width_col,
                  pad_h, pad_w, grad_im.data_ptr<float>(),
                  grad_offset.data_ptr<float>(), grad_mask.data_ptr<float>());
        };
        if (fused_y) launch(std::true_type{}); else launch(std::false_type{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// PER-ELEMENT col2im, smem-shared geometry. grad_im/offset/mask PRE-ZEROED.
void modulated_deformable_col2im_perelem_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3 &&
                  stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
                  dilation_w == 1 && channels % 32 == 0 && channels <= 1024,
              "perelem_smem col2im: dg==1, 3x3, s1, d1, channels%32==0, <=1024");
  dim3 grid(width_col, height_col, batch_size * kernel_h * kernel_w);
  dim3 block(channels);
  const bool fused_y = (data_offset.size(-1) == 3 * kernel_h * kernel_w);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_col2im_perelem_smem_nhwc", ([&] {
        auto launch = [&](auto FU) {
          dcn_nhwc::modulated_deformable_col2im_perelem_smem_nhwc_gpu_kernel<
              scalar_t, 3, 3, decltype(FU)::value>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  dcol.data_ptr<scalar_t>(), data_im.data_ptr<scalar_t>(),
                  data_offset.data_ptr<scalar_t>(), data_mask.data_ptr<scalar_t>(),
                  batch_size, channels, height_im, width_im, height_col, width_col,
                  pad_h, pad_w, grad_im.data_ptr<float>(),
                  grad_offset.data_ptr<float>(), grad_mask.data_ptr<float>());
        };
        if (fused_y) launch(std::true_type{}); else launch(std::false_type{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

// ATOMIC-FREE grad_X via gather. win = search half-window (>=2). gX PRE-ZEROED
// not required (direct write), but pass a zeroed/empty [B,H,W,Cin] fp32 tensor.
void modulated_deformable_gradx_gather_nhwc_cuda(
    const Tensor dcol, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, const int win,
    Tensor grad_im) {
  TORCH_CHECK(deformable_group == 1 && kernel_h == 3 && kernel_w == 3 &&
                  stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
                  dilation_w == 1 && channels % 32 == 0,
              "gather grad_X: dg==1, 3x3, stride1, dil1, channels%32==0");
  const int NCG = channels / 32;
  const int threads = 256;
  const int wpb = threads / dcn_nhwc::WARP_SIZE;
  dim3 grid((width_im + wpb - 1) / wpb, height_im, batch_size * NCG);
  dim3 block(threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dcol.scalar_type(),
      "modulated_deformable_gradx_gather_nhwc", ([&] {
        auto launch = [&](auto WIN) {
          dcn_nhwc::modulated_deformable_col2im_gradx_gather_nhwc_gpu_kernel<
              scalar_t, 3, 3, decltype(WIN)::value>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                  dcol.data_ptr<scalar_t>(), data_offset.data_ptr<scalar_t>(),
                  data_mask.data_ptr<scalar_t>(), batch_size, channels,
                  height_im, width_im, height_col, width_col, pad_h, pad_w, NCG,
                  grad_im.data_ptr<float>());
        };
        if (win <= 2) launch(std::integral_constant<int, 2>{});
        else if (win == 3) launch(std::integral_constant<int, 3>{});
        else launch(std::integral_constant<int, 4>{});
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
