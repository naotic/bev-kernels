#pragma once

// Max warps per block for the channels32 col2im backward kernel.
// Per-warp shared geometry arrays are sized by this; the launch block size
// must satisfy (blockDim.x / 32) <= MSDA_COL2IM_MAX_WARPS.
#ifndef MSDA_COL2IM_MAX_WARPS
#define MSDA_COL2IM_MAX_WARPS 8
#endif

#define MSDA_1D_KERNEL_LOOP(i, n)                                 \
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);       \
       i += blockDim.x * gridDim.x)

inline int msda_get_blocks(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}
