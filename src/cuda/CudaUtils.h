#pragma once

#include "Constants.h"

namespace ai
{
    namespace cuda
    {
        // For benchmarking.
        std::vector<int32_t> FindUniquesCPU(const std::vector<int32_t>& src);
        std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src);
    }
}
