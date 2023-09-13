#include "pch.h"

#include "Utils.h"

#include "cuda/CudaUtils.h"

const size_t ARRAY_SIZE = 100;
const unsigned MIN_UNIQUE = 10;
const unsigned MAX_UNIQUE = 50;

int main() {
    // Init random source array.
    auto srcArray = ai::utils::GenerateArray(ARRAY_SIZE, MIN_UNIQUE, MAX_UNIQUE);
    ai::utils::PrintArray("Source array", srcArray, 10);

    auto uniques = ai::cuda::FindUniquesGPU(srcArray);
    ai::utils::PrintArray("Uniques", uniques, 2);

    std::cin.get();
    return 0;
}
