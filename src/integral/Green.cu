#include <cstdio>

#include <cuda_runtime.h>

#include "integral/Constant.cuh"
#include "integral/Green.cuh"


namespace integral
{

namespace
{

constexpr float kDistEps {1e-9f};

__constant__
float2 center[kCenterUnrollFactor];

}  // namespace anomynous


cudaError_t setCenterAsync(
        const float2 * __restrict__ dCenter,
        int centerLen,
        cudaStream_t stream  // = nullptr in Green.cuh
)
{
    return cudaMemcpyToSymbolAsync(
            center,
            dCenter,
            sizeof(float2) * centerLen,
            0U,
            cudaMemcpyDeviceToDevice,
            stream
    );
}


__global__
void laplacianGreenFunction(
        const float2 * __restrict__ sample,
        int sampleLen,
        int segmentLenPadded,
        int centerLen,
        float laplacian,
        float scale,
        float * __restrict__ res
)
{
    auto blockSize = static_cast<int>(blockDim.x * blockDim.y);
    auto threadIdxInBlock = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    auto idx = static_cast<int>(blockIdx.x * kBlockUnrollFactor * blockSize + threadIdxInBlock);

    if (idx < sampleLen)
    {
        float2 r[kBlockUnrollFactor] = {};
        int riMax = 0;

        for (int si = idx; riMax != kBlockUnrollFactor and si < sampleLen; ++riMax, si += blockSize)
        {
            r[riMax] = sample[si];
        }

        auto resIdx = static_cast<int>(blockIdx.x * blockSize + threadIdxInBlock);

        #pragma unroll
        for (int ci = 0; ci < centerLen; ++ci)
        {
            float2 c = center[ci];

            float tmp = 0.0f;

            for (int ri = 0; ri != riMax; ++ri)
            {
                if (float r2 = (r[ri].x - c.x) * (r[ri].x - c.x) + (r[ri].y - c.y) * (r[ri].y - c.y); kDistEps < r2)
                {
                    tmp += 0.25f * laplacian * M_1_PIf32 * log(r2) * scale;
                }
            }

            res[segmentLenPadded * ci + resIdx] = tmp;

#ifdef DEBUG_GREENS_FUNCTION
            // printf("G(x=%f, y=%f, x0=%f, y0=%f) r2 = %f\n", x, y, x0, y0, r2);
            // if (batch == 0)  printf("i = %d, sampleLen = %d\n", i, sampleLen);
            // printf("\t\tsample[%d] and sample[%d], before setting res[%d] = %f\n", i, i + 1, i >> 1U, res[i >> 1U]);
#endif  // DEBUG_GREENS_FUNCTION
        }
    }
}


__global__
void laplacianGreenFunctionGradient(
        const float2 * __restrict__ sample,
        int sampleLen,
        int segmentLenPadded,
        int centerLen,
        float laplacian,
        float scale,
        float2 * __restrict__ grad
)
{
    auto blockSize = static_cast<int>(blockDim.x * blockDim.y);
    auto threadIdxInBlock = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    auto idx = static_cast<int>(blockIdx.x * kBlockUnrollFactor * blockSize + threadIdxInBlock);

    if (idx < sampleLen)
    {
        float2 r[kBlockUnrollFactor] = {};
        int riMax = 0;

        for (int si = idx; riMax != kBlockUnrollFactor and si < sampleLen; ++riMax, si += blockSize)
        {
            r[riMax] = sample[si];
        }

        auto resIdx = static_cast<int>(blockIdx.x * blockSize + threadIdxInBlock);

        #pragma unroll
        for (int ci = 0; ci < centerLen; ++ci)
        {
            float2 c = center[ci];

            float2 tmp = {0.0f, 0.0f};

            for (int ri = 0; ri != riMax; ++ri)
            {
                if (float r2 = (r[ri].x - c.x) * (r[ri].x - c.x) + (r[ri].y - c.y) * (r[ri].y - c.y); kDistEps < r2)
                {
                    float base = 0.5f * laplacian * M_1_PIf32 / r2 * scale;
                    tmp.x += base * (c.x - r[ri].x);
                    tmp.y += base * (c.y - r[ri].y);
                }
            }

            grad[segmentLenPadded * ci + resIdx] = tmp;
        }
    }
}

}  // namespace pte
