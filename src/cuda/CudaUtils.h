#pragma once

namespace ai
{
    namespace cuda
    {
        std::vector<int32_t> FindUniquesCPU(const std::vector<int32_t>& src, size_t uniqueSize);
        std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src, size_t uniqueSize);
    }
}
