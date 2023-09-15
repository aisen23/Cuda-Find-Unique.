#pragma once

#include <limits>

/** Input data
 * */
namespace ai
{

    const unsigned ARRAY_SIZE = 10000000;
    const unsigned MIN_UNIQUE = 10;
    const unsigned MAX_UNIQUE = 1000;

    const int32_t MIN_NUM = -100000;
    const int32_t MAX_NUM = 100000;
    const int32_t INVALID_NUM = std::numeric_limits<int32_t>::min();

} // ai
