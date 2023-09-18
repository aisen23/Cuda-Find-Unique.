#pragma once

#include <limits.h>

/** Input data
 * */
namespace ai
{

    const unsigned ARRAY_SIZE = 10000000;
    const unsigned MIN_UNIQUE = 10;
    const unsigned MAX_UNIQUE = 1000;

    const int32_t MIN_NUM = std::numeric_limits<int32_t>::min();
    // Maximum is reserved for an invalid big number.
    const int32_t MAX_NUM = std::numeric_limits<int32_t>::max() - 1;
    const int32_t MAX_INT_32 = std::numeric_limits<int32_t>::max();

    const unsigned CUDA_BLOCK_SIZE = 1024;

} // ai
