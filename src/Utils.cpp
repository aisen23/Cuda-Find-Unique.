#include "pch.h"

#include "Utils.h"
#include "Constants.h"

std::vector<int32_t> ai::utils::GenerateArray(size_t size, unsigned minUnique, unsigned maxUnique) {
    bool generateArrayValidArgs = 0 < minUnique && minUnique < maxUnique && maxUnique < size;
    assert(generateArrayValidArgs);
    if (!generateArrayValidArgs) {
        return {};
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate size of unique numbers array.
    std::uniform_int_distribution<unsigned> uDist(minUnique, maxUnique);
    size_t uSize = static_cast<size_t>(uDist(gen));

    // Generating numbers (unique and non-unique):
    auto [uniques, uniqueSet] = GenerateUniques(uSize);
    auto nonUniques = GenerateNonUniques(size - uSize, uniqueSet);


    // Fill array:
    std::vector<int32_t> array_(size, ai::INVALID_NUM);
    std::uniform_int_distribution<size_t> indexDist(0, size - 1);
    for (size_t i = 0; i != uSize;) {
        auto index = indexDist(gen);
        if (array_[index] == ai::INVALID_NUM) {
            array_[index] = uniques[i];
            ++i;
        } 
    }

    for (size_t i = 0, j = 0; i != size && j != nonUniques.size(); ++i) {
        if (array_[i] == ai::INVALID_NUM) {
            array_[i] = nonUniques[j++];
        }
    }

    return array_;
}
        
std::pair<std::vector<int32_t>, std::unordered_set<int32_t>> ai::utils::GenerateUniques(size_t size) {
    bool generateUniqueValidArgs = size > 0;
    assert(generateUniqueValidArgs );
    if (!generateUniqueValidArgs ) {
        return {};
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(ai::MIN_NUM, ai::MAX_NUM);

    std::vector<int32_t> uniques;
    uniques.reserve(size);

    std::unordered_set<int32_t> uniqueSet;

    for (size_t i = 0; i != size; ) {
        int32_t num = dist(gen);
        if (uniqueSet.find(num) == uniqueSet.end()) {
            uniques.push_back(num);
            uniqueSet.insert(num);
            ++i;
        }
    }

    return std::make_pair(std::move(uniques), std::move(uniqueSet));
}
        
std::vector<int32_t> ai::utils::GenerateNonUniques(size_t size, const std::unordered_set<int32_t>& uniqueSet) {
    bool generateNonUniqueValidArgs = size > 10;
    assert(generateNonUniqueValidArgs );
    if (!generateNonUniqueValidArgs ) {
        return {};
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(ai::MIN_NUM, ai::MAX_NUM);
    std::uniform_int_distribution<unsigned> partsDist(2, 4);

    unsigned partsCount = partsDist(gen);
    size_t nSize = size / partsCount;
    std::vector<int32_t> nonUnique;
    nonUnique.reserve(size);

    for (size_t i = 0; i != nSize; ) {
        int32_t num = dist(gen);
        if (uniqueSet.find(num) == uniqueSet.end()) {
            nonUnique.push_back(num);
            ++i;
        }
    }

    // Make numbers a non-unique.
    for (size_t i = 0; i != nSize; ++i) {
        nonUnique.push_back(nonUnique[i]);
    }
    
    std::shuffle(nonUnique.begin(), nonUnique.begin() + nSize, gen);

    std::uniform_int_distribution<size_t> indexDist(0, nonUnique.size() - 1);
    while (nonUnique.size() < size) {
        size_t index = indexDist(gen);
        nonUnique.push_back(nonUnique[index]);
    }

    return nonUnique;
}


void ai::utils::PrintArray(const char* name, const std::vector<int32_t>& array_, size_t num) {
    std::cout << name << ": ";

    size_t size = array_.size();
    if (size == 0) {
        return;
    }

    assert(num < size);

    for (size_t i = 0; i < num; ++i) {
        std::cout << array_[i] << ", ";
    }
    std::cout << "........";

    for (size_t i = size / 2; i < size / 2 + num; ++i) {
        std::cout << array_[i] << ", ";
    }
    std::cout << "........";

    for (size_t i = size - num; i < size; ++i) {
        std::cout << array_[i] << ", ";
    }
    std::cout << "\nsize: " << size << "\n\n";
}
