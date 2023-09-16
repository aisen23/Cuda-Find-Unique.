#include "pch.h"

#include "clock/Clock.h"
#include "Constants.h"
#include "threads/ThreadPool.h"
#include "Utils.h"

#include "cuda/CudaUtils.h"

int main() {
    ai::ThreadPool::Instance();
    // Init random source array.
    ai::Clock clock;

//+_+_+_-=-=- =- Array generation  =- =- == = -=- =- =- =- 

    std::cout << "int32_t MIN: " << std::numeric_limits<int32_t>::min() << std::endl;
    std::cout << "int32_t MAX: " << std::numeric_limits<int32_t>::max() << std::endl;

    auto arrayGenStart = clock.Now();

    auto srcArray = ai::utils::GenerateArray(ai::ARRAY_SIZE, ai::MIN_UNIQUE, ai::MAX_UNIQUE);
    
    std::cout << "Generating: "; clock.PrintDuration(arrayGenStart);
    ai::utils::PrintArray("Source array", srcArray, 10);


// -=-= -=- =-= Looking for uniques -= -=- =-=- =- =-= -=- =- =- =- =- =-= -

    auto searchUniquesStart = clock.Now();

    auto uniques = ai::cuda::FindUniquesGPU(srcArray);

    std::cout << "Computation Uniques: "; clock.PrintDuration(searchUniquesStart);
#ifdef DEBUG_BUILD
    auto testU = uniques;
    std::sort(testU.begin(), testU.end());
    ai::utils::PrintArray("Generated uniques", testU, 5);
#else
    ai::utils::PrintArray("Uniques", uniques, 10);
#endif


// -=- =-= -=- =- =-  -= -- = -= -= -=- =- 

    std::cin.get();
    return 0;
}
