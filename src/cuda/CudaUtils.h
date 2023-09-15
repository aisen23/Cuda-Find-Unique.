#pragma once

namespace ai
{
    namespace cuda
    {
        std::vector<int32_t> FindUniquesCPU(const std::vector<int32_t>& src);
        std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src);
    }
}
