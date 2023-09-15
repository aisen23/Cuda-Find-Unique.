#include "pch.h"

#include "clock/Clock.h"
#include "threads/ThreadPool.h"
#include "Utils.h"

#include "cuda/CudaUtils.h"

const size_t ARRAY_SIZE = 10000000;
const unsigned MIN_UNIQUE = 10;
const unsigned MAX_UNIQUE = 1000;

int main() {
    ai::ThreadPool::Instance();
    // Init random source array.
    ai::Clock clock;

//+_+_+_-=-=- =- Array generation  =- =- == = -=- =- =- =- 

    std::cout << "int32_t MIN: " << std::numeric_limits<int32_t>::min() << std::endl;
    std::cout << "int32_t MAX: " << std::numeric_limits<int32_t>::max() << std::endl;

    auto arrayGenStart = clock.Now();

    auto srcArray = ai::utils::GenerateArray(ARRAY_SIZE, MIN_UNIQUE, MAX_UNIQUE);
    
    std::cout << "Generating: "; clock.PrintDuration(arrayGenStart);
    ai::utils::PrintArray("Source array", srcArray, 10);


// -=-= -=- =-= Looking for uniques -= -=- =-=- =- =-= -=- =- =- =- =- =-= -

    auto searchUniquesStart = clock.Now();

    auto uniques = ai::cuda::FindUniquesGPU(srcArray, MAX_UNIQUE);

    std::cout << "Computation Uniques: "; clock.PrintDuration(searchUniquesStart);
    ai::utils::PrintArray("Uniques", uniques, 10);


// -=- =-= -=- =- =-  -= -- = -= -= -=- =- 

    std::cin.get();
    return 0;
}
