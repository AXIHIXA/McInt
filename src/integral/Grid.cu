#include "integral/Grid.cuh"


namespace integral
{

__global__
void generateSquareGridSamples(
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float step,
        float2 * __restrict__ sample
)
{
    auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto idy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    float x = xMin + step * static_cast<float>(idx);
    float y = yMin + step * static_cast<float>(idy);

    if (xMin <= x and x <= xMax and yMin <= y and y <= yMax)
    {
        int nx = static_cast<int>((xMax - xMin) / step) + 1;
        int sampleIdx = idy * nx + idx;
        sample[sampleIdx].x = x;
        sample[sampleIdx].y = y;
    }
}

}  // namespace integral
