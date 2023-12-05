#include <cstdio>

#include <cuda_runtime.h>

#include "integral/Geometry.cuh"


namespace integral
{

namespace
{

__forceinline__ __device__
float cross(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}


__forceinline__ __device__
float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}


__forceinline__ __device__
float2 operator-(float2 a, float2 b)
{
    return {a.x - b.x, a.y - b.y};
}


constexpr float kCrossEps {1e-3f};


// Winding number of a polygon:
// Non-zero if strictly inside;
// 0 if strictly outside;
// Inconsistant if on edge.
__device__
void windingNumber(
        const float2 * __restrict__ v,
        int vLen,
        float2 s,
        bool * __restrict__ onEdge,
        int * __restrict__ w
)
{
    *onEdge = false;
    *w = 0;

    float2 p = v[vLen - 1];
    float2 d;

    for (int i = 0; i < vLen; ++i)
    {
        d = v[i];

        float cs = cross(p - s, d - s);

        if (abs(cs) < kCrossEps and dot(p - s, d - s) < kCrossEps)
        {
            *onEdge = true;
            return;
        }

        if (p.y <= s.y)
        {
            if (s.y < d.y and kCrossEps < cs)
            {
                ++(*w);
            }
        }
        else
        {
            if (d.y <= s.y and cs < -kCrossEps)
            {
                --(*w);
            }
        }

        p = d;
    }
}


__device__
void pointInPolygonTest(
        const float2 * __restrict__ vert,
        const int * __restrict__ vertPtr,
        int vertPtrLen,
        float2 s,
        bool * __restrict__ flag
)
{
    bool onEdge = false;
    int w = 0;
    windingNumber(vert, vertPtr[1], s, &onEdge, &w);

    if (onEdge)
    {
        *flag = true;
        return;
    }
    else if (w == 0)
    {
        *flag = false;
        return;
    }

    for (int i = 2; i != vertPtrLen; ++i)
    {
        windingNumber(vert + vertPtr[i - 1], vertPtr[i] - vertPtr[i - 1], s, &onEdge, &w);

        if (onEdge)
        {
            *flag = true;
            return;
        }
        else if (w != 0)
        {
            *flag = false;
            return;
        }
    }

    *flag = true;
}

}  // namespace anomynous


__global__
void insideTest(
        const float2 * __restrict__ vert,
        const int * __restrict__ vertPtr,
        int vertPtrLen,
        const float2 * __restrict__ sample,
        int numSamples,
        bool * __restrict__ mask
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < numSamples)
    {
        float2 s = sample[i];

        bool flag;
        pointInPolygonTest(vert, vertPtr, vertPtrLen, s, &flag);

        mask[i] = flag;
    }
}


__global__
void insideTestWithBox(
        const float2 * __restrict__ vert,
        const int * __restrict__ vertPtr,
        int vertPtrLen,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        const float2 * __restrict__ point,
        int pointLen,
        bool * __restrict__ mask
)
{
    auto i = static_cast<int>(blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x);

    if (i < pointLen)
    {
        float2 s = point[i];

        if (s.x < xMin or xMax < s.x or s.y < yMin or yMax < s.y)
        {
            mask[i] = false;
            return;
        }

        bool flag;
        pointInPolygonTest(vert, vertPtr, vertPtrLen, s, &flag);

        mask[i] = flag;
    }
}

}  // namespace pte
