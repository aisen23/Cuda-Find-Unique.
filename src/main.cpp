#include "pch.h"

#include "clock/Clock.h"
#include "Utils.h"

#include "cuda/CudaUtils.h"

const size_t ARRAY_SIZE = 10000000;
const unsigned MIN_UNIQUE = 10;
const unsigned MAX_UNIQUE = 1000;

int main() {
    // Init random source array.
    Clock clock;

//+_+_+_-=-=- =- Array generation  =- =- == = -=- =- =- =- 

    auto arrayGenStart = clock.Now();

    auto srcArray = ai::utils::GenerateArray(ARRAY_SIZE, MIN_UNIQUE, MAX_UNIQUE);
    
    clock.PrintDurationFrom(arrayGenStart);
    ai::utils::PrintArray("Source array", srcArray, 10);


// -=-= -=- =-= Looking for uniques -= -=- =-=- =- =-= -=- =- =- =- =- =-= -

    auto searchUniquesStart = clock.Now();

    auto uniques = ai::cuda::FindUniquesGPU(srcArray);

    clock.PrintDurationFrom(searchUniquesStart);
    ai::utils::PrintArray("Uniques", uniques, 2);


// -=- =-= -=- =- =-  -= -- = -= -= -=- =- 

    std::cin.get();
    return 0;
}
