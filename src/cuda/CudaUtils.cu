#include "pch.h"

#include "clock/Clock.h"
#include "CudaUtils.h"
#include "threads/ThreadPool.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ai::cuda
{
    __global__ void FindUniquesKernel(int32_t* array_, uint32_t size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size) {
            array_[tid] += 1;
        }
    }

    std::vector<int32_t> FindUniquesCPU(const std::vector<int32_t>& src) {

        std::unordered_map<int32_t, uint32_t> counter;
        std::unordered_set<int32_t> uniqueSet;

        for (size_t i = 0; i != src.size(); ++i) {
            auto num = src[i];
            int count = ++counter[num];

            if (count > 1) {
                uniqueSet.erase(num);
            }
            else {
                uniqueSet.insert(num);
            }
        }

        std::vector<int32_t> result; 
        result.reserve(uniqueSet.size());
        for (auto it = uniqueSet.begin(); it != uniqueSet.end(); ++it) {
            result.push_back(*it);
        }

        return result;
    }


    std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src) {
        uint32_t hostSize= src.size();
        const int32_t* hostData = src.data();
        int32_t* deviceData;

        std::vector<int32_t> result(hostSize);

        Clock clock;
        auto mallocStart = clock.Now();
        cudaMalloc(&deviceData, hostSize * sizeof(int32_t));
        std::cout << "Malloc: "; clock.PrintDuration(mallocStart);

        auto memcpyStart = clock.Now();
        cudaMemcpy(deviceData, hostData, hostSize * sizeof(int32_t), cudaMemcpyHostToDevice);
        std::cout << "Memcpy: "; clock.PrintDuration(memcpyStart);

        /*size_t numThreads = 20;
        size_t chunkSize = hostSize / numThreads;
        std::vector<std::future<void>> futures(numThreads - 1);
        for (size_t i = 0; i != numThreads - 1; ++i) {
            futures[i] = ThreadPool::Instance().Submit([deviceData, hostData, chunkSize, i]() {
                    auto offsettedData = deviceData + chunkSize * i;
                    cudaMemcpy(offsettedData, hostData + chunkSize * i, chunkSize * sizeof(int32_t), cudaMemcpyHostToDevice);
            });
        }
        auto offsettedData = deviceData + chunkSize * (numThreads - 1);
        cudaMemcpy(offsettedData, hostData + chunkSize * (numThreads - 1), chunkSize * sizeof(int32_t), cudaMemcpyHostToDevice);

        for (auto& f : futures) {
            f.wait();
        }*/

        const int size = 10000000;
        const int blockSize = 1024;
        const int gridSize = (size + blockSize - 1) / blockSize;

        auto cudaComp = clock.Now();
        FindUniquesKernel<<<gridSize, blockSize>>>(deviceData, hostSize);
        std::cout << "CudaComputation: "; clock.PrintDuration(cudaComp);

        auto memcpyFree = clock.Now();
        cudaMemcpy(result.data(), deviceData, size * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(deviceData);
        std::cout << "CudaMemcpyFree: "; clock.PrintDuration(memcpyFree);

        return result;
    }
}
