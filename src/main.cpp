#include "pch.h"

#include "clock/Clock.h"
#include "Compressor.h"
#include "DataUtils.h"

const size_t ARRAY_SIZE = 1000000;

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Init random source array.
    auto srcArray = ai::utils::CreateAnArray(ARRAY_SIZE);


    // Real test (and example) for compressing and uncompressing.
    std::vector<uint8_t> compressed;
    std::vector<int> decompressedArray;

    {   
        ai::Clock clock;
        {
            auto startCompressTime = clock.Now();

            ai::Compressor compressor(ai::eCompressorType::Compressor);
            compressed = compressor.Compress(srcArray);

            std::cout << "Compressing duration: ";
            clock.PrintDurationFrom(startCompressTime);
        }

        {
            auto startDecompressTime = clock.Now();

            ai::Compressor uncompressor(ai::eCompressorType::Uncompressor);
            decompressedArray = uncompressor.Uncompress(compressed);

            std::cout << "Decompressing duration: ";
            clock.PrintDurationFrom(startDecompressTime);
            std::cout << "\n";
        }
    }



    // Benchmarking: 
    std::cout << "Source array size: " << sizeof(int) * srcArray.size() << std::endl;
    std::cout << "Compressed size: " << sizeof(uint8_t) * compressed.size() << "\n\n";

    ai::utils::PrintArray("Source array", srcArray);

    ai::utils::PrintArray("Compressed array", compressed);

    ai::utils::PrintArray("Uncompressed array", decompressedArray);

    std::cin.get();
    return 0;
}
