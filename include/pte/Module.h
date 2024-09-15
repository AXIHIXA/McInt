#ifndef MODULE_H
#define MODULE_H

#include <torch/torch.h>


namespace pte
{

pybind11::tuple iint(
        torch::Tensor & vert,
        torch::Tensor & vertPtr,
        torch::Tensor & center,
        float f
);


torch::Tensor geometryMask(
        torch::Tensor & vert,
        torch::Tensor & vertPtr,
        torch::Tensor & point
);

}  // namespace pte


#endif  // MODULE_H
