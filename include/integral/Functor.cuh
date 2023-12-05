#ifndef FUNCTOR_H
#define FUNCTOR_H

#include <thrust/random.h>
#include <thrust/tuple.h>


namespace integral
{

class UniformFloat2
{
public:
    UniformFloat2() = delete;

    UniformFloat2(unsigned int seed, float xMin, float xMax, float yMin, float yMax)
            : e(seed), dx(xMin, xMax), dy(yMin, yMax)
    {
        // Nothing needed here.
    }

    __host__ __device__ float2 operator()(unsigned long long i)
    {
        e.discard(i);
        return {dx(e), dy(e)};
    }

private:
    thrust::default_random_engine e;

    thrust::uniform_real_distribution<float> dx;

    thrust::uniform_real_distribution<float> dy;
};


struct ZeroMask
{
    template <typename Tuple>
    __host__ __device__
    bool operator()(Tuple t)
    {
        return thrust::get<1>(t) == false;
    }
};


struct AddFloat2
{
    __host__ __device__
    float2 operator()(const float2 & a, const float2 & b)
    {
        return {a.x + b.x, a.y + b.y};
    }
};


template <typename Scalar>
struct MultiplyByScalar
{
    explicit MultiplyByScalar(Scalar factor = 1.0f)
        : factor(factor)
    {
        // Nothing Needed here
    }

    __host__ __device__
    Scalar operator()(Scalar scalar)
    {
        return factor * scalar;
    }

    Scalar factor;
};

}  // namespace pte


#endif  // FUNCTOR_H
