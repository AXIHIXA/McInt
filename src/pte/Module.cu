#include "integral/Integral.h"
#include "pte/Module.h"


namespace pte
{

pybind11::tuple iint(
        torch::Tensor & vert,
        torch::Tensor & vertPtr,
        torch::Tensor & center,
        float f
)
{
    auto [vertMin, vertMinIndices] = torch::min(vert, 0);
    auto [vertMax, vertMaxIndices] = torch::max(vert, 0);

//    torch::Tensor vertMinCpu = vertMin.cpu();
//    torch::Tensor vertMaxCpu = vertMax.cpu();

    torch::Tensor res = torch::empty(
            {center.size(0)},
            torch::TensorOptions().dtype(torch::kFloat32).device(center.device())
    );

    if (vert.requires_grad())
    {
        torch::Tensor grad = torch::empty_like(
            vert,
            torch::TensorOptions().dtype(torch::kFloat32).device(vert.device())
        );

        integral::iint(
                reinterpret_cast<float2 *>(vert.data_ptr<float>()),
                vertPtr.data_ptr<int>(),
                static_cast<int>(vertPtr.size(0)),
                vertMin.index({0}).item<float>(),
                vertMax.index({0}).item<float>(),
                vertMin.index({1}).item<float>(),
                vertMax.index({1}).item<float>(),
                reinterpret_cast<float2 *>(center.data_ptr<float>()),
                static_cast<int>(center.size(0)),
                f,
                res.data_ptr<float>(),
                reinterpret_cast<float2 *>(grad.data_ptr<float>())
        );

        return pybind11::make_tuple(res, grad);
    }
    else
    {
        integral::iint(
                reinterpret_cast<float2 *>(vert.data_ptr<float>()),
                vertPtr.data_ptr<int>(),
                static_cast<int>(vertPtr.size(0)),
                vertMin.index({0}).item<float>(),
                vertMax.index({0}).item<float>(),
                vertMin.index({1}).item<float>(),
                vertMax.index({1}).item<float>(),
                reinterpret_cast<float2 *>(center.data_ptr<float>()),
                static_cast<int>(center.size(0)),
                f,
                res.data_ptr<float>(),
                nullptr
        );

        return pybind11::make_tuple(res, pybind11::none());
    }
}


torch::Tensor geometryMask(
        torch::Tensor & vert,
        torch::Tensor & vertPtr,
        torch::Tensor & point
)
{
    auto [vertMin, vertMinIndices] = torch::min(vert, 0);
    auto [vertMax, vertMaxIndices] = torch::max(vert, 0);

    torch::Tensor mask = torch::empty(
            {point.size(0)},
            torch::TensorOptions().dtype(torch::kInt).device(point.device())
    );

    integral::insidePolygonTest(
            reinterpret_cast<float2 *>(vert.data_ptr<float>()),
            vertPtr.data_ptr<int>(),
            static_cast<int>(vertPtr.size(0)),
            vertMin.index({0}).item<float>(),
            vertMax.index({0}).item<float>(),
            vertMin.index({1}).item<float>(),
            vertMax.index({1}).item<float>(),
            reinterpret_cast<float2 *>(point.data_ptr<float>()),
            static_cast<int>(point.size(0)),
            mask.data_ptr<int>()
    );

    return mask;
}

}  // namespace pte
