#include "pch.h"

#include "CudaUtils.h"

#include "clock/Clock.h"
#include "Constants.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace ai::cuda
{
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

        std::sort(result.begin(), result.end());

        return result;
    }

//=-=-=-=-=-=-=-=-=--=-= GPU -=-=-=-=-=-=-=-=-=--=-=-=-=-=-

    __device__ void SetBlockFlag(uint32_t* blockFlags) {
        int id = blockIdx.x / 32;

        blockFlags[id] |= (1 << (blockIdx.x % 32));
    }

    __device__ bool IsBlockFlag(unsigned blockId, const uint32_t* blockFlags) {
        int id = blockId / 32;
        return (blockFlags[id] & (1 << (blockId % 32)));
    }

    // Wait until neighbors temp will be filled.
    __device__ void WaitForNeighbors(const uint32_t* blockFlags) {
        if (blockIdx.x < gridDim.x) {
            while (!((blockIdx.x == 0 || IsBlockFlag(blockIdx.x - 1, blockFlags))
                    && (blockIdx.x == blockDim.x - 1 || IsBlockFlag(blockIdx.x + 1, blockFlags))));
        }
    }

    __global__ void FindUniquesKernel(int32_t* array_, uint32_t size, uint32_t* blockFlags) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        __shared__ int32_t temp[ai::CUDA_BLOCK_SIZE];

        bool unique = true;

        if (tid < size) {
            if (tid > 0) {
                if (array_[tid] == array_[tid - 1]) {
                    unique = false;
                }
            }

            if (unique && tid < size - 1) {
                if (array_[tid] == array_[tid + 1]) {
                    unique = false;
                }
            }

            temp[threadIdx.x] = (unique ? array_[tid] : ai::MAX_INT_32);
        }

        __syncthreads();
        
        if (threadIdx.x == 0) {
            SetBlockFlag(blockFlags);
        }
        
        if (tid < size) {
            WaitForNeighbors(blockFlags);

            array_[tid] = temp[threadIdx.x];
        }
    }

    __global__ void FindUniquesSizeKernel(int32_t* array_, uint32_t size, uint32_t* uSize) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid > 0 && tid < size) {
            if (array_[tid] == ai::MAX_INT_32 && array_[tid - 1] != ai::MAX_INT_32) {
                *uSize = tid;
            }
        }
    }


    std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src) {
        //Clock clock;

        // Preparing Data.
       // auto preparingTime = clock.Now();

        const uint32_t arraySize = src.size();
        thrust::device_vector<int32_t> dArray(arraySize);
        thrust::copy(src.begin(), src.end(), dArray.begin());

        // Device uniques size.
        uint32_t* dUSize;
        auto cudaStatus = cudaMalloc(&dUSize, sizeof(uint32_t));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return {};
        }
        cudaMemset(dUSize, 0, sizeof(uint32_t));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(dUSize);
            return {};
        }

        // For synchronizing the neighbor blocks.
        int blockSize = ai::CUDA_BLOCK_SIZE;
        int gridSize = (arraySize + blockSize - 1) / blockSize;
        uint32_t* dBlockFlags;
        uint32_t blockFlagsSize = (gridSize + 32 - 1) / 32 * sizeof(uint32_t);
        cudaStatus = cudaMalloc(&dBlockFlags, blockFlagsSize);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(dUSize);
            return {};
        }
        cudaStatus = cudaMemset(dBlockFlags, 0, blockFlagsSize);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(dUSize);
            cudaFree(dBlockFlags);
            return {};
        }

        //std::cout << "Preparing: "; clock.PrintDuration(preparingTime);


        // Computation.
        //auto compTime = clock.Now();

        thrust::sort(dArray.begin(), dArray.end());

        FindUniquesKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(dArray.data()), arraySize, dBlockFlags);
        cudaDeviceSynchronize();

        thrust::sort(dArray.begin(), dArray.end());


        FindUniquesSizeKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(dArray.data()), arraySize, dUSize);
        cudaDeviceSynchronize();

        uint32_t uniquesSize;
        cudaStatus = cudaMemcpy(&uniquesSize, dUSize, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(dUSize);
            cudaFree(dBlockFlags);
            return {};
        }

        std::vector<int32_t> uniques(uniquesSize);
        thrust::copy(dArray.begin(), dArray.begin() + uniquesSize, uniques.begin());

        //std::cout << "Computation: "; clock.PrintDuration(compTime);


        // Free memory.
        cudaFree(dUSize);
        cudaFree(dBlockFlags);

        return uniques;
    }
}
