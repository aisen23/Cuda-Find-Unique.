#include "pch.h"

#include "Utils.h"

std::vector<int32_t> ai::utils::GenerateArray(size_t size, unsigned minUnique, unsigned maxUnique) {
    bool generateArrayValidArgs = 0 < minUnique && minUnique < maxUnique && maxUnique < size;
    assert(generateArrayValidArgs);
    if (!generateArrayValidArgs) {
        return {};
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<unsigned> uDist(minUnique, maxUnique);
    size_t uniqueSize = static_cast<size_t>(uDist(gen));

    std::vector<int32_t> uniqueNumbers;
    std::unordered_set<int32_t> uniqueNumbersSet;

    // Generating numbers (unique and non-unique):
    auto [uniques, uniqueSet] = GenerateUniques(uniqueSize);
    auto nonUniques = GenerateNonUniques(size - uniqueSize, uniqueSet);


    // Fill array:
    std::vector<int32_t> array_(size);
    std::unordered_set<size_t> uniquePlaces;
    std::uniform_int_distribution<size_t> indexDist(0, size - 1);
    for (auto u : uniques) {
        auto index = indexDist(gen);
        if (uniquePlaces.find(index) == uniquePlaces.end()) {
            array_[index] = u;
            uniquePlaces.insert(index);
        } 
    }

    for (size_t i = 0, j = 0; i != size && j != nonUniques.size(); ++i) {
        if (uniquePlaces.find(i) == uniquePlaces.end()) {
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

    int32_t minInt = std::numeric_limits<int32_t>::min();
    int32_t maxInt = std::numeric_limits<int32_t>::max();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(minInt, maxInt);

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

    return std::make_pair(uniques, uniqueSet);
}
        
std::vector<int32_t> ai::utils::GenerateNonUniques(size_t size, const std::unordered_set<int32_t>& uniqueSet) {
    bool generateNonUniqueValidArgs = size > 10;
    assert(generateNonUniqueValidArgs );
    if (!generateNonUniqueValidArgs ) {
        return {};
    }

    int32_t minInt = std::numeric_limits<int32_t>::min();
    int32_t maxInt = std::numeric_limits<int32_t>::max();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(minInt, maxInt);
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
