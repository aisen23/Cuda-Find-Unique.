#include "pch.h"

#include "Utils.h"

std::vector<int32_t> ai::utils::GenerateArray(size_t size, unsigned minUnique, unsigned maxUnique) {
    std::random_device rd;
    std::mt19937 gen(rd());
    int32_t _min = 0;
    int32_t _max = 10000;
    std::uniform_int_distribution<int32_t> dist(_min, _max);

    std::unordered_set<int32_t> arraysNums;
    unsigned uniqueCount = 0;

    std::vector<int32_t> array_;
    
    while (array_.size() < size) {
        int32_t num = dist(gen);
        auto it = arraysNums.find(num);
        bool inArray = (it != arraysNums.end());

        if ((inArray && uniqueCount <= minUnique) || (!inArray && uniqueCount >= maxUnique)) {
            continue;
        }

        if (!inArray) {
            arraysNums.insert(num);
            ++uniqueCount;
        }
        else {
            --uniqueCount;
        }

        array_.push_back(num);
    }

    return array_;
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
