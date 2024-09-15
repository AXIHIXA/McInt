#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

namespace integral
{

//__global__
//void insideTest(
//        const float2 * __restrict__ vert,
//        const int * __restrict__ vertPtr,
//        int vertPtrLen,
//        const float2 * __restrict__ sample,
//        int numSamples,
//        bool * __restrict__ mask
//);


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
        int * __restrict__ mask
);

}  // namespace pte


#endif  // GEOMETRY_CUH
