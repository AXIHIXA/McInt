#include <random>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include "integral/Constant.cuh"
#include "integral/Functor.cuh"
#include "integral/Geometry.cuh"
#include "integral/Integral.h"
#include "integral/Green.cuh"
#include "integral/Grid.cuh"
#include "util/CudaUtil.h"


namespace integral
{

namespace
{

inline int padToNk(int a, int n)
{
    if (int r = a % n; r == 0)
    {
        return a;
    }
    else
    {
        return a + n - r;
    }
}


void laplacianGreenFunctionIntegral(
        const float2 * __restrict__ sample,
        int sampleLen,
        const float2 * __restrict__ center,
        int centerLen,
        float f,
        float scale,
        float * __restrict__ res
)
{
    // Try unrolling w.r.t. centers.
    // I.e., Each CUDA thread takes one sample point, but deals with multiple centers.
    // Output format in dBuffer:
    // ----------------- Index i ---------------->
    //        center0               center1
    // [sample0 ... sampleN] [sample0 ... sampleN] ...

    // Pad numSamples to multiple of 32
    // (32 * sizeof(float) == 128, L1 cache line granularity)
    // for aligned & coalesced memory access pattern.
    int numSamplesPerBlock = kBlockSize * kBlockUnrollFactor;
    unsigned int numBlocks = (sampleLen + numSamplesPerBlock - 1) / numSamplesPerBlock;
    int segmentLen = static_cast<int>(numBlocks * kBlockSize);
    int segmentLenPadded = padToNk(segmentLen, 32);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float> dBuffer(segmentLenPadded * kCenterUnrollFactor, 0.0f);
    dim3 intergralGridDim {numBlocks, 1U, 1U};

    // Unrolling segmentation for cub::DeviceSegmentedReduce::Sum.
    // We have TWO different segmentation:
    // (1) For preceeding full-sized batches (dBuffer contains result for kUnrollLevel centers);
    // (2) For the last batch (dBuffer contains result for less-than kUnrollLevel centers, plus succeeding trash data).
    int centerLenPerBatch = kCenterUnrollFactor;
    int centerLenLastBatch = centerLen % centerLenPerBatch;

    thrust::device_vector<int> dBeginOffset(centerLen);
    thrust::device_vector<int> dEndOffset(centerLen);
    thrust::sequence(thrust::device, dBeginOffset.begin(), dBeginOffset.end(), 0, segmentLenPadded);
    thrust::sequence(thrust::device, dEndOffset.begin(), dEndOffset.end(), segmentLen, segmentLenPadded);

    // CUBReduction Configuration
    std::size_t tempStorageBytes = 0UL;
    CUDA_CHECK(
            cub::DeviceSegmentedReduce::Sum(
                    nullptr,
                    tempStorageBytes,
                    dBuffer.data().get(),
                    res,
                    centerLenPerBatch,
                    dBeginOffset.data().get(),
                    dEndOffset.data().get(),
                    nullptr
            )
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);

    cudaStream_t s1;
    cudaStream_t s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    for (int ci = 0, step = kCenterUnrollFactor; ci < centerLen; ci += step)
    {
        int centerLenThisBatch =
                centerLenLastBatch != 0 and centerLen <= ci + step ?
                centerLenLastBatch : centerLenPerBatch;

        // thrust::fill(thrust::device, dBuffer.begin(), dBuffer.end(), 0.0f);
        CUDA_CHECK(cudaMemsetAsync(dBuffer.data().get(), 0, sizeof(float) * dBuffer.size(), s1));
        CUDA_CHECK(setCenterAsync(center + ci, centerLenThisBatch, s2));

        laplacianGreenFunction<<<intergralGridDim, kBlockDim, 0U, nullptr>>>(
                sample,
                sampleLen,
                segmentLenPadded,
                centerLenThisBatch,
                f,
                scale,
                dBuffer.data().get()
        );
        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(
                cub::DeviceSegmentedReduce::Sum(
                        dTempStorage.data().get(),
                        tempStorageBytes,
                        dBuffer.data().get(),
                        res + ci,
                        centerLenThisBatch,
                        dBeginOffset.data().get(),
                        dEndOffset.data().get(),
                        nullptr
                )
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
}


void laplacianGreenFunctionIntegralGradient(
        const float2 * __restrict__ sample,
        int sampleLen,
        const float2 * __restrict__ center,
        int centerLen,
        float f,
        float scale,
        float2 * __restrict__ grad
)
{
    // Try unrolling w.r.t. centers.
    // I.e., Each CUDA thread takes one sample point, but deals with multiple centers.
    // Output format in dBuffer:
    // ----------------- Index i ---------------->
    //        center0               center1
    // [sample0 ... sampleN] [sample0 ... sampleN] ...

    // Pad numSamples to multiple of 16
    // (16 * sizeof(float2) == 128, L1 cache line granularity)
    // for aligned & coalesced memory access pattern.
    int numSamplesPerBlock = kBlockSize * kBlockUnrollFactor;
    unsigned int numBlocks = (sampleLen + numSamplesPerBlock - 1) / numSamplesPerBlock;
    int segmentLen = static_cast<int>(numBlocks * kBlockSize);
    int segmentLenPadded = padToNk(segmentLen, 16);
    // int numSamplesPadded = numSamples;  // Test for non-aligned pattern. Slower!
    thrust::device_vector<float2> dBuffer(segmentLenPadded * kCenterUnrollFactor, {0.0f, 0.0f});
    dim3 intergralGridDim {numBlocks, 1U, 1U};

    // Unrolling segmentation for cub::DeviceSegmentedReduce::Sum.
    // We have TWO different segmentation:
    // (1) For preceeding full-sized batches (dBuffer contains result for kUnrollLevel centers);
    // (2) For the last batch (dBuffer contains result for less-than kUnrollLevel centers, plus succeeding trash data).
    int centerLenPerBatch = kCenterUnrollFactor;
    int centerLenLastBatch = centerLen % centerLenPerBatch;

    thrust::device_vector<int> dBeginOffset(centerLen);
    thrust::device_vector<int> dEndOffset(centerLen);
    thrust::sequence(thrust::device, dBeginOffset.begin(), dBeginOffset.end(), 0, segmentLenPadded);
    thrust::sequence(thrust::device, dEndOffset.begin(), dEndOffset.end(), segmentLen, segmentLenPadded);

    // CUBReduction Configuration
    std::size_t tempStorageBytes = 0UL;
    CUDA_CHECK(
            cub::DeviceSegmentedReduce::Reduce(
                    nullptr,
                    tempStorageBytes,
                    dBuffer.data().get(),
                    grad,
                    centerLenPerBatch,
                    dBeginOffset.data().get(),
                    dEndOffset.data().get(),
                    AddFloat2(),
                    float2 {0.0f, 0.0f},
                    nullptr
            )
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_vector<unsigned char> dTempStorage(tempStorageBytes);

    cudaStream_t s1;
    cudaStream_t s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    for (int ci = 0, step = kCenterUnrollFactor; ci < centerLen; ci += step)
    {
        int centerLenThisBatch =
                centerLenLastBatch != 0 and centerLen <= ci + step ?
                centerLenLastBatch : centerLenPerBatch;

        CUDA_CHECK(cudaMemsetAsync(dBuffer.data().get(), 0, sizeof(float2) * dBuffer.size(), s1));
        CUDA_CHECK(setCenterAsync(center + ci, centerLenThisBatch, s2));

        laplacianGreenFunctionGradient<<<intergralGridDim, kBlockDim, 0U, nullptr>>>(
                sample,
                sampleLen,
                segmentLenPadded,
                centerLenThisBatch,
                f,
                scale,
                dBuffer.data().get()
        );
        CUDA_CHECK_LAST_ERROR();

        CUDA_CHECK(
                cub::DeviceSegmentedReduce::Reduce(
                        dTempStorage.data().get(),
                        tempStorageBytes,
                        dBuffer.data().get(),
                        grad + ci,
                        centerLenThisBatch,
                        dBeginOffset.data().get(),
                        dEndOffset.data().get(),
                        AddFloat2(),
                        float2 {0.0f, 0.0f},
                        nullptr
                )
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
}

}  // namespace anomynous


void iint(
        const float2 * __restrict__ vert,
        const int * __restrict__ vertPtr,
        int vertPtrLen,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        const float2 * __restrict__ center,
        int centerLen,
        float f,
        float * __restrict__ res,
        float2 * __restrict__ grad
)
{
    constexpr int imageSize = 129;
    constexpr int pixelGranularity = 16;
    constexpr float step = 1.0f / static_cast<float>(pixelGranularity);
    xMin = 0.0f;
    xMax = imageSize;
    yMin = 0.0f;
    yMax = imageSize;

    int initialNumSamples = imageSize * imageSize * pixelGranularity * pixelGranularity;
    thrust::device_vector<float2> dSample(initialNumSamples);

    dim3 sampleGenerationGridDim {
        static_cast<unsigned int>(imageSize * pixelGranularity + kBlockDim.x - 1U) / kBlockDim.x,
        static_cast<unsigned int>(imageSize * pixelGranularity + kBlockDim.y - 1U) / kBlockDim.y,
        1U
    };
    generateSquareGridSamples<<<sampleGenerationGridDim, kBlockDim>>>(
            xMin, xMax, yMin, yMax, step, dSample.data().get()
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

//    unsigned int seed = std::random_device()();
//
//    thrust::transform(
//            thrust::device,
//            thrust::make_counting_iterator(0LL),
//            thrust::make_counting_iterator(kNumSamples),
//            dSample.begin(),
//            UniformFloat2(seed, xMin, xMax, yMin, yMax)
//    );

//    {
//        thrust::device_vector<bool> dMask(kNumSamples, false);
//
//        dim3 insideTestGridDim {static_cast<unsigned int>(kNumSamples + kBlockSize - 1U) / kBlockSize, 1U, 1U};
//
//        insideTest<<<insideTestGridDim, kBlockDim, 0U, nullptr>>>(
//                vert, vertPtr, vertPtrLen,
//                dSample.data().get(), kNumSamples,
//                dMask.data().get()
//        );
//        CUDA_CHECK_LAST_ERROR();
//        CUDA_CHECK(cudaDeviceSynchronize());
//
//        auto [dSampleIt, itPk] = thrust::remove_if(
//                thrust::device,
//                thrust::make_zip_iterator(dSample.begin(), dMask.begin()),
//                thrust::make_zip_iterator(dSample.end(), dMask.end()),
//                ZeroMask()
//        ).get_iterator_tuple();
//        dSample.erase(dSampleIt, dSample.end());
//    }

    float boxArea = (xMax - xMin) * (yMax - yMin);
    float scale = boxArea / static_cast<float>(initialNumSamples);

#ifdef DEBUG_INSIDE_MASK
    std::printf(
            "dSample.size() = %zu, Area = %lf\n",
            dSample.size(),
            static_cast<double>(dSample.size()) / static_cast<double>(kNumSamples) * boxArea
    );
#endif  // DEBUG_INSIDE_MASK

    laplacianGreenFunctionIntegral(
            dSample.data().get(),
            static_cast<int>(dSample.size()),
            center,
            centerLen,
            f,
            scale,
            res
    );

    if (grad)
    {
        int vertLen;
        CUDA_CHECK(cudaMemcpy(&vertLen, vertPtr + vertPtrLen - 1, sizeof(int), cudaMemcpyDeviceToHost));

        // TODO: Handle exterior/inner convex/concave cases.
        laplacianGreenFunctionIntegralGradient(
                dSample.data().get(),
                static_cast<int>(dSample.size()),
                vert,
                vertLen,
                f,
                scale,
                grad
        );
    }
}


void insidePolygonTest(
        const float2 * __restrict__ vert,
        const int * __restrict__ vertPtr,
        int vertPtrLen,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        const float2 * __restrict__ point,
        int pointLen,
        int * __restrict__ mask
)
{
    dim3 insideTestGridDim {static_cast<unsigned int>(pointLen + kBlockSize - 1U) / kBlockSize, 1U, 1U};

    insideTestWithBox<<<insideTestGridDim, kBlockDim, 0U, nullptr>>>(
            vert, vertPtr, vertPtrLen,
            xMin, xMax, yMin, yMax,
            point, pointLen,
            mask
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace pte
