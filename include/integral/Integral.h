#ifndef INTEGRAL_H
#define INTEGRAL_H


namespace integral
{

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
);


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
        bool * __restrict__ mask
);

}  // namespace pte


#endif  // INTEGRAL_H
