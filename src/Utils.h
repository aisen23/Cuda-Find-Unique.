#pragma once

#include <vector>
#include <unordered_set>

#include<cstdint>

namespace ai
{
    namespace utils
    {
        std::vector<int32_t> GenerateArray(size_t size, unsigned min, unsigned max);
        std::pair<std::vector<int32_t>, std::unordered_set<int32_t>> GenerateUniques(size_t size);
        std::vector<int32_t> GenerateNonUniques(size_t size, const std::unordered_set<int32_t>& uniqueSet);

        void PrintArray(const char* name, const std::vector<int32_t>& array_, size_t num); 
    } // utils
} // ai
