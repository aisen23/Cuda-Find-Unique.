#include "pch.h"

#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ai::cuda
{
    __global__ void myTest() {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        printf("CUDA thread %d is here!\n", tid);
    }


    std::vector<int32_t> FindUniquesGPU(const std::vector<int32_t>& src) {
        
        myTest<<<1, 1>>>();
        cudaDeviceSynchronize();

        return {-1, -1, -1, -1};
    }
}
