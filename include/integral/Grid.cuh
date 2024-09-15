#ifndef GRID_CUH
#define GRID_CUH

#include <cuda_runtime.h>


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
);

}  // namespace integral


#endif  // GRID_CUH
