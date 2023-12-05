#ifndef INTEGRALKERNEL_H
#define INTEGRALKERNEL_H


namespace integral
{

/// @brief
/// Evaluate f * G over an sample (represented by 0.5 * sampleLen 2D points),
/// where G Green's function for Laplacian operators,
/// G(x, y, x0, y0) = ln sqrt((x - x0)**2 + (y - y0)**2),
/// and f is the Laplacian at sample point.
/// (x0, y0)s are stored in __constant__ float center[kUnrollLevel].
/// @param sample          ...
/// @param sampleLen       ...
/// @param segmentLenPadded segmentLen padded to multiple of 32 to fit 128-Byte L1 cache line granularity.
/// @param laplacian       ...
/// @param res             ...
__global__
void laplacianGreenFunction(
        const float2 * __restrict__ sample,
        int sampleLen,
        int segmentLenPadded,
        int centerLen,
        float laplacian,
        float scale,
        float * __restrict__ res
);


__global__
void laplacianGreenFunctionGradient(
        const float2 * __restrict__ sample,
        int sampleLen,
        int segmentLenPadded,
        int centerLen,
        float laplacian,
        float scale,
        float2 * __restrict__ grad
);


cudaError_t setCenterAsync(
        const float2 * __restrict__ dCenter,
        int centerLen,
        cudaStream_t stream = nullptr
);

}  // namespace pte


#endif  // INTEGRALKERNEL_H
