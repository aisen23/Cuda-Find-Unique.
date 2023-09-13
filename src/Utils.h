#pragma once

#include <vector>

#include<cstdint>

namespace ai
{
    namespace utils
    {
        std::vector<int32_t> GenerateArray(size_t size, unsigned min, unsigned max);

        void PrintArray(const char* name, const std::vector<int32_t>& array_, size_t num); 
    } // utils
} // ai
