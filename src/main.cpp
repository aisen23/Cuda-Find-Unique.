#include "pch.h"

#include "clock/Clock.h"
#include "Constants.h"
#include "Utils.h"

#include "cuda/CudaUtils.h"

int main() {
    // Init random source array.
    ai::Clock clock;

//+_+_+_-=-=- =- Array generation  =- =- == = -=- =- =- =- 

    std::cout << "MIN: " << ai::MIN_NUM << "\n";
    std::cout << "MAX: " << ai::MAX_NUM << "\n\n";

    auto arrayGenStart = clock.Now();

    auto srcArray = ai::utils::GenerateArray(ai::ARRAY_SIZE, ai::MIN_UNIQUE, ai::MAX_UNIQUE);
    
    std::cout << "Generating: "; clock.PrintDuration(arrayGenStart);
    ai::utils::PrintArray("Source array", srcArray, 10);


// -=-= -=- =-= Looking for uniques -= -=- =-=- =- =-= -=- =- =- =- =- =-= -

    auto searchUniquesStart = clock.Now();

    auto uniques = ai::cuda::FindUniquesGPU(srcArray);

    std::cout << "Computation Uniques: "; clock.PrintDuration(searchUniquesStart);
    ai::utils::PrintArray("Uniques", uniques, 10);


// -=- =-= -=- =- =-  -= -- = -= -= -=- =- 

    std::cin.get();
    return 0;
}
