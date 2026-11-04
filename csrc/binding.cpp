#include <torch/extension.h>

using at::Tensor;

void modulated_deformable_im2col_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_im2col_opt_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_opt_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_smem_cuda(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_col2im_coord_opt_cuda(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

// --- NHWC fused path (dg==1) ---
void modulated_deformable_im2col_nhwc_cuda(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_fused_im2col_nhwc_cuda(
    const Tensor data_im, const Tensor w_off, const Tensor b_off,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_col2im_coord_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_offset,
    Tensor grad_mask);

void modulated_deformable_gradx_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_gradx_gather_nhwc_cuda(
    const Tensor dcol, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, const int win,
    Tensor grad_im);

void modulated_deformable_col2im_fused_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_col2im_perelem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask);

void modulated_deformable_col2im_perelem_smem_nhwc_cuda(
    const Tensor dcol, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im,
    Tensor grad_offset, Tensor grad_mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("modulated_deformable_im2col_cuda",
        &modulated_deformable_im2col_cuda,
        "Modulated deformable im2col (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_cuda",
        &modulated_deformable_col2im_cuda,
        "Modulated deformable col2im (CUDA/MUSA)");
  m.def("modulated_deformable_im2col_opt_cuda",
        &modulated_deformable_im2col_opt_cuda,
        "Optimized modulated deformable im2col (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_opt_cuda",
        &modulated_deformable_col2im_opt_cuda,
        "Optimized modulated deformable col2im (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_smem_cuda",
        &modulated_deformable_col2im_smem_cuda,
        "Shared-memory tiled col2im (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_coord_cuda",
        &modulated_deformable_col2im_coord_cuda,
        "Modulated deformable col2im_coord (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_coord_opt_cuda",
        &modulated_deformable_col2im_coord_opt_cuda,
        "Optimized modulated deformable col2im_coord (CUDA/MUSA)");
  m.def("modulated_deformable_im2col_nhwc_cuda",
        &modulated_deformable_im2col_nhwc_cuda,
        "Modulated deformable im2col gather, NHWC dg=1 (CUDA/MUSA)");
  m.def("modulated_deformable_fused_im2col_nhwc_cuda",
        &modulated_deformable_fused_im2col_nhwc_cuda,
        "Fused F1-F5 offset-conv + gather + modulation, NHWC dg=1 (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_nhwc_cuda",
        &modulated_deformable_col2im_nhwc_cuda,
        "Backward col2im (grad_im/offset/mask), NHWC dg=1 (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_coord_nhwc_cuda",
        &modulated_deformable_col2im_coord_nhwc_cuda,
        "Backward grad_offset/grad_mask only, NHWC dg=1 (CUDA/MUSA)");
  m.def("modulated_deformable_gradx_smem_nhwc_cuda",
        &modulated_deformable_gradx_smem_nhwc_cuda,
        "Backward grad_X via shared-memory tiling, NHWC 3x3 s1 (CUDA/MUSA)");
  m.def("modulated_deformable_gradx_gather_nhwc_cuda",
        &modulated_deformable_gradx_gather_nhwc_cuda,
        "Backward grad_X via atomic-free gather, NHWC 3x3 s1 (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_fused_smem_nhwc_cuda",
        &modulated_deformable_col2im_fused_smem_nhwc_cuda,
        "Fully-fused B4-B9 col2im w/ smem-tiled grad_X, NHWC 3x3 s1 (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_perelem_nhwc_cuda",
        &modulated_deformable_col2im_perelem_nhwc_cuda,
        "Per-element B4-B9 col2im (TileLang-style), NHWC 3x3 s1 (CUDA/MUSA)");
  m.def("modulated_deformable_col2im_perelem_smem_nhwc_cuda",
        &modulated_deformable_col2im_perelem_smem_nhwc_cuda,
        "Per-element col2im w/ smem-shared geometry, NHWC 3x3 s1 (CUDA/MUSA)");
}
