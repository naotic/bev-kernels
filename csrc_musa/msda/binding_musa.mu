#include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
#include <ATen/musa/MUSAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "ms_deform_attn_kernels.cuh"

static constexpr int kSpecHeads = 8;
static constexpr int kSpecChannels = 32;

// Env-gated col2im backward block size (H20-style smem-reduce occupancy boost).
// Default OFF -> 32 threads (1 warp/block), byte-identical to baseline launch.
// MSDA_COL2IM_SMEM_REDUCE=1 -> 256 threads (8 warps/block); each warp still owns
// its own (b,q,head) work-unit, with per-warp shared geometry slots. Math is
// unchanged; only atomicAdd ordering differs (fp32 tolerance).
static int msda_col2im_block_threads() {
  // Cache the env lookup so we don't getenv on every backward call.
  static const int threads = []() -> int {
    const char *e = std::getenv("MSDA_COL2IM_SMEM_REDUCE");
    if (e == nullptr)
      return 32;
    const int v = std::atoi(e);
    if (v <= 0)
      return 32;
    // 256 threads = 8 warps; must not exceed MSDA_COL2IM_MAX_WARPS warps.
    const int max_threads = MSDA_COL2IM_MAX_WARPS * 32;
    int t = 256;
    if (t > max_threads)
      t = max_threads;
    return t;
  }();
  return threads;
}

static inline bool is_spec_a(int num_heads, int channels, int num_levels,
                             int num_point) {
  return num_heads == kSpecHeads && channels == kSpecChannels &&
         num_levels == 1 && num_point == 4;
}

static inline bool is_spec_b(int num_heads, int channels, int num_levels,
                             int num_point) {
  return num_heads == kSpecHeads && channels == kSpecChannels &&
         num_levels == 4 && num_point == 8;
}

// SpecA: keep warp4 spec kernel (NUM_LEVELS=1, NUM_POINTS=4).

template <typename scalar_t>
void launch_im2col_generic(musaStream_t stream, const int n,
                           const scalar_t *data_value,
                           const int64_t *data_spatial_shapes,
                           const int64_t *data_level_start_index,
                           const scalar_t *data_sampling_loc,
                           const scalar_t *data_attn_weight, int batch_size,
                           int spatial_size, int num_heads, int channels,
                           int num_levels, int num_query, int num_point,
                           scalar_t *data_col) {
  const int threads = 512;
  const int blocks = msda_get_blocks(n, threads);
  at::musa::FastDivmod fastdivmod_channels(channels);
  at::musa::FastDivmod fastdivmod_num_heads(num_heads);
  at::musa::FastDivmod fastdivmod_num_query(num_query);
  ms_deformable_im2col_generic_kernel<scalar_t>
      <<<blocks, threads, 0, stream>>>(
          n, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, data_col,
          fastdivmod_channels, fastdivmod_num_heads, fastdivmod_num_query);
}

static void launch_im2col_spec_a(musaStream_t stream, const float *data_value,
                                 const int64_t *data_spatial_shapes,
                                 const int64_t *data_level_start_index,
                                 const float *data_sampling_loc,
                                 const float *data_attn_weight, int batch_size,
                                 int spatial_size, int num_query,
                                 float *data_col) {
  const int units = batch_size * num_query * kSpecHeads;
  // 固化为最优已验证变体：warp4_shm_ptr
  const int threads = 128;
  const int blocks = (units + 4 - 1) / 4;
  // SpecA forward kernel: NUM_LEVELS=1, NUM_POINTS=4.
  // Geometry precompute path is disabled for this instantiation,
  // so we only need the original shared buffer for (loc_w/loc_h/weight).
  const size_t shmem = static_cast<size_t>(4) * (1 * 4) * 3 * sizeof(float);
  ms_deform_im2col_spec_forward_f32_warp4_shm_ptr_kernel<1, 4>
      <<<blocks, threads, shmem, stream>>>(
          data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, spatial_size, num_query,
          batch_size, data_col);
}

// SpecB: warp4 spec kernel (NUM_LEVELS=4, NUM_POINTS=8).
static void launch_im2col_spec_b(musaStream_t stream, const float *data_value,
                                  const int64_t *data_spatial_shapes,
                                  const int64_t *data_level_start_index,
                                  const float *data_sampling_loc,
                                  const float *data_attn_weight,
                                  int batch_size, int spatial_size,
                                  int num_query, float *data_col) {
  const int units = batch_size * num_query * kSpecHeads;
  const int threads = 128;  // warp4
  const int blocks = (units + 4 - 1) / 4;
  // spec forward kernel shared layout:
  // loc: (NUM_LEVELS*NUM_POINTS*3) floats
  // geom: +10 floats worth of scalars (WarpGeomI32 = 5 floats + 5 ints => 40B)
  constexpr int kNumLevels = 4;
  constexpr int kNumPoints = 8;
  constexpr int kWarpCount = 4;  // threads/32
  const size_t shmem =
      static_cast<size_t>(kWarpCount) *
      ((kNumLevels * kNumPoints * 3 + 10) * sizeof(float));

  ms_deform_im2col_spec_forward_f32_warp4_shm_ptr_kernel<kNumLevels, kNumPoints>
      <<<blocks, threads, shmem, stream>>>(
          data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, spatial_size, num_query,
          batch_size, data_col);
}

// fp16 forward launch: SpecA (L=1,P=4) and SpecB (L=4,P=8) share the same
// __half kernel template; geometry / gate logic mirrors the f32 spec kernel.
static void launch_im2col_spec_f16(musaStream_t stream, int num_levels,
                                   int num_point, const __half *data_value,
                                   const int64_t *data_spatial_shapes,
                                   const int64_t *data_level_start_index,
                                   const __half *data_sampling_loc,
                                   const __half *data_attn_weight,
                                   int batch_size, int spatial_size,
                                   int num_query, __half *data_col) {
  const int units = batch_size * num_query * kSpecHeads;
  const int threads = 128;  // warp4
  const int blocks = (units + 4 - 1) / 4;
  constexpr int kWarpCount = 4;  // threads/32
  if (num_levels == 1 && num_point == 4) {
    const size_t shmem =
        static_cast<size_t>(kWarpCount) * (1 * 4 * 3) * sizeof(float);
    ms_deform_im2col_spec_forward_f16_warp4_shm_ptr_kernel<1, 4>
        <<<blocks, threads, shmem, stream>>>(
            data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, spatial_size, num_query,
            batch_size, data_col);
  } else {  // num_levels == 4 && num_point == 8
    const size_t shmem =
        static_cast<size_t>(kWarpCount) * (4 * 8 * 3) * sizeof(float);
    ms_deform_im2col_spec_forward_f16_warp4_shm_ptr_kernel<4, 8>
        <<<blocks, threads, shmem, stream>>>(
            data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, spatial_size, num_query,
            batch_size, data_col);
  }
}

// fp16 col2im launch: grad_value points to an fp32 accumulation buffer.
static void launch_col2im_spec_f16(
    musaStream_t stream, int num_levels, int num_point, const int n,
    const __half *grad_col, int64_t g_stride_b, int64_t g_stride_q,
    int64_t g_stride_m, int64_t g_stride_c, bool grad_contig,
    const __half *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const __half *data_sampling_loc,
    const __half *data_attn_weight, int batch_size, int spatial_size,
    int num_heads, int channels, int num_query, float *grad_value_f32,
    __half *grad_sampling_loc, __half *grad_attn_weight) {
  const int threads = msda_col2im_block_threads();
  const int blocks = msda_get_blocks(n, threads);
  at::musa::FastDivmod fastdivmod_channels(channels);
  at::musa::FastDivmod fastdivmod_num_heads(num_heads);
  at::musa::FastDivmod fastdivmod_num_query(num_query);
  if (num_levels == 1 && num_point == 4) {
    ms_deform_col2im_channels32_warp_reduce_f16_kernel_spec<1, 4>
        <<<blocks, threads, 0, stream>>>(
            n, grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c,
            grad_contig, data_value, data_spatial_shapes,
            data_level_start_index, data_sampling_loc, data_attn_weight,
            batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, num_point, grad_value_f32, grad_sampling_loc,
            grad_attn_weight, fastdivmod_channels, fastdivmod_num_heads,
            fastdivmod_num_query);
  } else {  // num_levels == 4 && num_point == 8
    ms_deform_col2im_channels32_warp_reduce_f16_kernel_spec<4, 8>
        <<<blocks, threads, 0, stream>>>(
            n, grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c,
            grad_contig, data_value, data_spatial_shapes,
            data_level_start_index, data_sampling_loc, data_attn_weight,
            batch_size, spatial_size, num_heads, channels, num_levels,
            num_query, num_point, grad_value_f32, grad_sampling_loc,
            grad_attn_weight, fastdivmod_channels, fastdivmod_num_heads,
            fastdivmod_num_query);
  }
}

template <typename scalar_t>
void launch_col2im(musaStream_t stream, const int n, const scalar_t *grad_col,
                   int64_t g_stride_b, int64_t g_stride_q, int64_t g_stride_m,
                   int64_t g_stride_c, bool grad_contig,
                   const scalar_t *data_value,
                   const int64_t *data_spatial_shapes,
                   const int64_t *data_level_start_index,
                   const scalar_t *data_sampling_loc,
                   const scalar_t *data_attn_weight, int batch_size,
                   int spatial_size, int num_heads, int channels,
                   int num_levels, int num_query, int num_point,
                   scalar_t *grad_value, scalar_t *grad_sampling_loc,
                   scalar_t *grad_attn_weight, bool force_generic) {
  const bool use_warp32 =
      !force_generic && channels == 32 && std::is_same<scalar_t, float>::value;

  if (use_warp32) {
    const int threads = msda_col2im_block_threads();
    const int blocks = msda_get_blocks(n, threads);
    at::musa::FastDivmod fastdivmod_channels(channels);
    at::musa::FastDivmod fastdivmod_num_heads(num_heads);
    at::musa::FastDivmod fastdivmod_num_query(num_query);
    if (num_levels == 1 && num_point == 4) {
      ms_deform_col2im_channels32_warp_reduce_f32_kernel_spec<1, 4><<<
          blocks, threads, 0, stream>>>(
          n, reinterpret_cast<const float *>(grad_col), g_stride_b, g_stride_q,
          g_stride_m, g_stride_c, grad_contig,
          reinterpret_cast<const float *>(data_value), data_spatial_shapes,
          data_level_start_index,
          reinterpret_cast<const float *>(data_sampling_loc),
          reinterpret_cast<const float *>(data_attn_weight), batch_size,
          spatial_size, num_heads, channels, num_levels, num_query, num_point,
          reinterpret_cast<float *>(grad_value),
          reinterpret_cast<float *>(grad_sampling_loc),
          reinterpret_cast<float *>(grad_attn_weight), fastdivmod_channels,
          fastdivmod_num_heads, fastdivmod_num_query);
    } else if (num_levels == 4 && num_point == 8) {
      ms_deform_col2im_channels32_warp_reduce_f32_kernel_spec<4, 8><<<
          blocks, threads, 0, stream>>>(
          n, reinterpret_cast<const float *>(grad_col), g_stride_b, g_stride_q,
          g_stride_m, g_stride_c, grad_contig,
          reinterpret_cast<const float *>(data_value), data_spatial_shapes,
          data_level_start_index,
          reinterpret_cast<const float *>(data_sampling_loc),
          reinterpret_cast<const float *>(data_attn_weight), batch_size,
          spatial_size, num_heads, channels, num_levels, num_query, num_point,
          reinterpret_cast<float *>(grad_value),
          reinterpret_cast<float *>(grad_sampling_loc),
          reinterpret_cast<float *>(grad_attn_weight), fastdivmod_channels,
          fastdivmod_num_heads, fastdivmod_num_query);
    } else {
      ms_deform_col2im_channels32_warp_reduce_f32_kernel<<<
          blocks, threads, 0, stream>>>(
          n, reinterpret_cast<const float *>(grad_col), g_stride_b, g_stride_q,
          g_stride_m, g_stride_c, grad_contig,
          reinterpret_cast<const float *>(data_value), data_spatial_shapes,
          data_level_start_index,
          reinterpret_cast<const float *>(data_sampling_loc),
          reinterpret_cast<const float *>(data_attn_weight), batch_size,
          spatial_size, num_heads, channels, num_levels, num_query, num_point,
          reinterpret_cast<float *>(grad_value),
          reinterpret_cast<float *>(grad_sampling_loc),
          reinterpret_cast<float *>(grad_attn_weight), fastdivmod_channels,
          fastdivmod_num_heads, fastdivmod_num_query);
    }
    return;
  }

  TORCH_CHECK(
      channels <= 1024,
      "msda_opt generic col2im fallback supports channels<=1024 in this build");

  const int threads = channels;
  const int blocks = msda_get_blocks(n, threads);
  const size_t shmem = static_cast<size_t>(threads) * 3 * sizeof(scalar_t);

  at::musa::FastDivmod fastdivmod_channels(channels);
  at::musa::FastDivmod fastdivmod_num_heads(num_heads);
  at::musa::FastDivmod fastdivmod_num_query(num_query);
  ms_deformable_col2im_shm_stride_v1_kernel<scalar_t>
      <<<blocks, threads, shmem, stream>>>(
          n, grad_col, g_stride_b, g_stride_q, g_stride_m, g_stride_c,
          grad_contig, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, grad_value,
          grad_sampling_loc, grad_attn_weight, fastdivmod_channels, fastdivmod_num_heads, fastdivmod_num_query);
}

at::Tensor msda_forward_musa(const at::Tensor &value,
                             const at::Tensor &spatial_shapes,
                             const at::Tensor &level_start_index,
                             const at::Tensor &sampling_loc,
                             const at::Tensor &attn_weight, int im2col_step,
                             bool force_generic) {
  TORCH_CHECK(value.is_musa(), "value must be MUSA");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(),
              "level_start_index must be contiguous");
  TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
  TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");

  const int batch = static_cast<int>(value.size(0));
  const int spatial_size = static_cast<int>(value.size(1));
  const int num_heads = static_cast<int>(value.size(2));
  const int channels = static_cast<int>(value.size(3));
  const int num_levels = static_cast<int>(spatial_shapes.size(0));
  const int num_query = static_cast<int>(sampling_loc.size(1));
  const int num_point = static_cast<int>(sampling_loc.size(4));

  const int im2col_step_ = std::min(batch, im2col_step);
  TORCH_CHECK(batch % im2col_step_ == 0, "batch must divide im2col_step");

  auto output =
      at::zeros({batch, num_query, num_heads * channels}, value.options());
  const int batch_n = im2col_step_;
  auto output_n = output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});
  const int64_t per_value_size =
      static_cast<int64_t>(spatial_size) * num_heads * channels;
  const int64_t per_sample_loc_size =
      static_cast<int64_t>(num_query) * num_heads * num_levels * num_point * 2;
  const int64_t per_attn_weight_size =
      static_cast<int64_t>(num_query) * num_heads * num_levels * num_point;

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // Forward dispatch policy:
  // - SpecA-real: keep spec forward.
  // - SpecB-real: enable spec forward (then keep generic fallback if force_generic=true).
  const bool use_spec_a_f32 = !force_generic && value.scalar_type() == at::kFloat &&
                               is_spec_a(num_heads, channels, num_levels, num_point);
  // SpecB-ish (num_query=2048) 的 spec kernel 当前比 generic 慢，因此只在更大 Q 时启用；
  // SpecB-real (num_query≈9504) 才启用以追求收益。
  constexpr int kSpecBMinNumQuery = 4096;
  const bool use_spec_b_f32 =
      !force_generic && value.scalar_type() == at::kFloat &&
      is_spec_b(num_heads, channels, num_levels, num_point) &&
      (num_query >= kSpecBMinNumQuery);

  // fp16 path: only the spec geometries (SpecA/SpecB, channels==32) are
  // supported by the __half kernel. Other half shapes fall through to the
  // generic float-typed dispatch below would fail (AT_DISPATCH has no half),
  // so we require the spec gate when value is half.
  const bool value_is_half = value.scalar_type() == at::kHalf;
  const bool use_spec_f16 =
      !force_generic && value_is_half &&
      (is_spec_a(num_heads, channels, num_levels, num_point) ||
       is_spec_b(num_heads, channels, num_levels, num_point));
  TORCH_CHECK(!value_is_half || use_spec_f16,
              "msda_opt2 fp16 forward only supports the SpecA/SpecB geometries "
              "(heads=8, channels=32, L/P in {1/4, 4/8}); got num_heads=",
              num_heads, " channels=", channels, " num_levels=", num_levels,
              " num_point=", num_point);

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    const int n_kernels = batch_n * num_query * num_heads * channels;

    if (use_spec_f16) {
      const __half *vptr =
          reinterpret_cast<const __half *>(value.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_value_size;
      const __half *lptr =
          reinterpret_cast<const __half *>(sampling_loc.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
      const __half *aptr =
          reinterpret_cast<const __half *>(attn_weight.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;
      __half *cptr = reinterpret_cast<__half *>(columns.data_ptr<at::Half>());
      launch_im2col_spec_f16(stream, num_levels, num_point, vptr,
                             spatial_shapes.data_ptr<int64_t>(),
                             level_start_index.data_ptr<int64_t>(), lptr, aptr,
                             batch_n, spatial_size, num_query, cptr);
      continue;
    }

    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "msda_forward_musa", [&] {
          const scalar_t *vptr = value.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_value_size;
          const scalar_t *lptr = sampling_loc.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
          const scalar_t *aptr = attn_weight.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;
          scalar_t *cptr = columns.data_ptr<scalar_t>();

          if (use_spec_a_f32 && std::is_same<scalar_t, float>::value) {
            launch_im2col_spec_a(
                stream, reinterpret_cast<const float *>(vptr),
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                reinterpret_cast<const float *>(lptr),
                reinterpret_cast<const float *>(aptr), batch_n, spatial_size,
                num_query, reinterpret_cast<float *>(cptr));
          } else if (use_spec_b_f32 && std::is_same<scalar_t, float>::value) {
            launch_im2col_spec_b(
                stream, reinterpret_cast<const float *>(vptr),
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                reinterpret_cast<const float *>(lptr),
                reinterpret_cast<const float *>(aptr), batch_n, spatial_size,
                num_query, reinterpret_cast<float *>(cptr));
          } else {
            launch_im2col_generic(stream, n_kernels, vptr,
                                  spatial_shapes.data_ptr<int64_t>(),
                                  level_start_index.data_ptr<int64_t>(), lptr,
                                  aptr, batch_n, spatial_size, num_heads,
                                  channels, num_levels, num_query, num_point,
                                  cptr);
          }
        });
  }
  return output;
}

void msda_backward_musa(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, int im2col_step, bool force_generic_col2im) {
  TORCH_CHECK(value.is_musa(), "value must be MUSA");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(),
              "level_start_index must be contiguous");
  TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
  TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");
  TORCH_CHECK(grad_output.is_musa(), "grad_output must be MUSA");

  const int batch = static_cast<int>(value.size(0));
  const int spatial_size = static_cast<int>(value.size(1));
  const int num_heads = static_cast<int>(value.size(2));
  const int channels = static_cast<int>(value.size(3));
  const int num_levels = static_cast<int>(spatial_shapes.size(0));
  const int num_query = static_cast<int>(sampling_loc.size(1));
  const int num_point = static_cast<int>(sampling_loc.size(4));

  const int im2col_step_ = std::min(batch, im2col_step);
  TORCH_CHECK(batch % im2col_step_ == 0, "batch must divide im2col_step");

  const int64_t per_value_size =
      static_cast<int64_t>(spatial_size) * num_heads * channels;
  const int64_t per_sample_loc_size =
      static_cast<int64_t>(num_query) * num_heads * num_levels * num_point * 2;
  const int64_t per_attn_weight_size =
      static_cast<int64_t>(num_query) * num_heads * num_levels * num_point;

  auto grad_5d = grad_output.view(
      {batch / im2col_step_, im2col_step_, num_query, num_heads, channels});

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // fp16 backward path: grad_value is accumulated in an internal fp32 buffer
  // (atomicAdd in float — the numerically-sensitive part), then narrowed into
  // the half grad_value output at the very end. grad_sampling_loc /
  // grad_attn_weight are written as half (reduced in float inside the kernel).
  const bool value_is_half = value.scalar_type() == at::kHalf;
  if (value_is_half) {
    const bool use_spec_f16 =
        !force_generic_col2im &&
        (is_spec_a(num_heads, channels, num_levels, num_point) ||
         is_spec_b(num_heads, channels, num_levels, num_point));
    TORCH_CHECK(use_spec_f16,
                "msda_opt2 fp16 backward only supports the SpecA/SpecB "
                "geometries (heads=8, channels=32, L/P in {1/4, 4/8})");

    auto grad_value_f32 =
        at::zeros_like(value, value.options().dtype(at::kFloat));

    for (int n = 0; n < batch / im2col_step_; ++n) {
      at::Tensor grad_chunk = grad_5d.select(0, n);
      const int n_kernels = im2col_step_ * num_query * num_heads * channels;
      const bool grad_contig = grad_chunk.is_contiguous();
      const int64_t sb = grad_chunk.stride(0);
      const int64_t sq = grad_chunk.stride(1);
      const int64_t sm = grad_chunk.stride(2);
      const int64_t sc = grad_chunk.stride(3);

      const __half *vptr =
          reinterpret_cast<const __half *>(value.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_value_size;
      const __half *gptr =
          reinterpret_cast<const __half *>(grad_chunk.data_ptr<at::Half>());
      const __half *lptr =
          reinterpret_cast<const __half *>(sampling_loc.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
      const __half *aptr =
          reinterpret_cast<const __half *>(attn_weight.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;
      float *gvptr_f32 = grad_value_f32.data_ptr<float>() +
                         static_cast<int64_t>(n) * im2col_step_ * per_value_size;
      __half *gslptr =
          reinterpret_cast<__half *>(grad_sampling_loc.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
      __half *gaptr =
          reinterpret_cast<__half *>(grad_attn_weight.data_ptr<at::Half>()) +
          static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;

      launch_col2im_spec_f16(
          stream, num_levels, num_point, n_kernels, gptr, sb, sq, sm, sc,
          grad_contig, vptr, spatial_shapes.data_ptr<int64_t>(),
          level_start_index.data_ptr<int64_t>(), lptr, aptr, im2col_step_,
          spatial_size, num_heads, channels, num_query, gvptr_f32, gslptr,
          gaptr);
    }
    // Narrow fp32 grad_value accumulator -> half grad_value output.
    grad_value.copy_(grad_value_f32);
    return;
  }

  for (int n = 0; n < batch / im2col_step_; ++n) {
    at::Tensor grad_chunk = grad_5d.select(0, n);
    const int n_kernels = im2col_step_ * num_query * num_heads * channels;
    const bool grad_contig = grad_chunk.is_contiguous();
    const int64_t sb = grad_chunk.stride(0);
    const int64_t sq = grad_chunk.stride(1);
    const int64_t sm = grad_chunk.stride(2);
    const int64_t sc = grad_chunk.stride(3);

    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "msda_backward_musa", [&] {
          const scalar_t *vptr = value.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_value_size;
          const scalar_t *gptr = grad_chunk.data_ptr<scalar_t>();
          const scalar_t *lptr = sampling_loc.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
          const scalar_t *aptr = attn_weight.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;
          scalar_t *gvptr = grad_value.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_value_size;
          scalar_t *gslptr = grad_sampling_loc.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_sample_loc_size;
          scalar_t *gaptr = grad_attn_weight.data_ptr<scalar_t>() +
              static_cast<int64_t>(n) * im2col_step_ * per_attn_weight_size;

          launch_col2im(
              stream, n_kernels, gptr, sb, sq, sm, sc, grad_contig, vptr,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(), lptr, aptr, im2col_step_,
              spatial_size, num_heads, channels, num_levels, num_query, num_point,
              gvptr, gslptr, gaptr, force_generic_col2im);
        });
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &msda_forward_musa, "MSDA forward", py::arg("value"),
        py::arg("spatial_shapes"), py::arg("level_start_index"),
        py::arg("sampling_loc"), py::arg("attn_weight"),
        py::arg("im2col_step") = 64, py::arg("force_generic") = false);
  m.def("backward", &msda_backward_musa, "MSDA backward", py::arg("value"),
        py::arg("spatial_shapes"), py::arg("level_start_index"),
        py::arg("sampling_loc"), py::arg("attn_weight"),
        py::arg("grad_output"), py::arg("grad_value"),
        py::arg("grad_sampling_loc"), py::arg("grad_attn_weight"),
        py::arg("im2col_step") = 64,
        py::arg("force_generic_col2im") = false);
}
