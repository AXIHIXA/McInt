#include <torch/extension.h>

#include "pte/Module.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using py::literals::operator""_a;

    m.def(
            "iint",
            pte::iint,
            "v"_a,
            "ptr"_a,
            "center"_a,
            "f"_a,
            py::return_value_policy::move
    );

    m.def(
            "inside",
            pte::inside,
            "v"_a,
            "ptr"_a,
            "point"_a,
            py::return_value_policy::move
    );
}
