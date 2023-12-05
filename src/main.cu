#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "integral/Integral.h"


int main(int argc, char * argv[])
{
    std::vector<float2> hVert
            {
#include "dat/vert.txt"
            };

    std::vector<int> hVertPtr
            {
#include "dat/vert_ptr.txt"
            };

    std::vector<float2> hCenter
            {
#include "dat/center.txt"
            };

    thrust::device_vector<float2> dVert = hVert;
    thrust::device_vector<int> dVertPtr = hVertPtr;
    thrust::device_vector<float2> dCenter = hCenter;
    thrust::device_vector<float> dRes(hCenter.size(), -123.0f);

    integral::iint(
            dVert.data().get(),
            dVertPtr.data().get(),
            static_cast<int>(dVertPtr.size()),
            2.0f,
            24.0f,
            2.0f,
            24.0f,
            dCenter.data().get(),
            //            static_cast<int>(dCenter.size()) << 1U,
            32 << 1U,  // Fewer centers, fewer kernel calls in loop, to falcilitate ncu replays
            -0.0002f,
            dRes.data().get()
    );

    return EXIT_SUCCESS;
}
