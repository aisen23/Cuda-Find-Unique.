#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
endif()
