#include "pch.h"

#include "clock/Clock.h"
#include "CudaUtils.h"
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

    std::vector<int32_t> FindUniquesCPU(const std::vector<int32_t>& src, size_t uniqueSize) {
        std::vector<int32_t> result(src.size());

        for (int i = 0; i != src.size(); ++i) {
            result[i] = src[i] + 1;
        }

        return result;
    }


    std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src, size_t uniqueSize) {
        uint32_t size = src.size();
        int32_t* deviceData;

        std::vector<int32_t> result(size);

        Clock clock;
        auto mallocMemcpy = clock.Now();
        cudaMalloc(&deviceData, size * sizeof(int32_t));
        cudaMemcpy(deviceData, src.data(), size * sizeof(int32_t), cudaMemcpyHostToDevice);
        std::cout << "MallocMemcpy: "; clock.PrintDurationFrom(mallocMemcpy);
        
        int totalThreads = size;
        int numThreads = 320;

        auto cudaComp = clock.Now();
        FindUniquesKernel<<<(totalThreads + numThreads - 1) / numThreads, numThreads>>>(deviceData, size);
        std::cout << "CudaComputation: "; clock.PrintDurationFrom(cudaComp);

        auto memcpyFree = clock.Now();
        cudaMemcpy(result.data(), deviceData, size * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(deviceData);
        std::cout << "CudaMemcpyFree: "; clock.PrintDurationFrom(memcpyFree);

        return result;
    }
}
