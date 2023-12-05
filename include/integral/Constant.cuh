#ifndef CONSTANT_CUH
#define CONSTANT_CUH

#include <cuda_runtime.h>


namespace integral
{

inline constexpr long long kNumSamples {10000000LL};

inline constexpr dim3 kBlockDim {32U, 32U, 1U};

inline constexpr int kBlockSize {static_cast<int>(kBlockDim.x * kBlockDim.y * kBlockDim.z)};

// Neighbor unrolling w.r.t. centers (loaded from constant memory)
inline constexpr int kCenterUnrollFactor {64};

// Block Unrolling w.r.t. samples (save global stores)
// 8: 179ms (but better compute/memory balance)
// 16: 127ms
inline constexpr int kBlockUnrollFactor {16};

}  // namespace integral


#endif  // CONSTANT_CUH
