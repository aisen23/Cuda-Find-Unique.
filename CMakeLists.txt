cmake_minimum_required(VERSION 3.27)

project(as_test_2 LANGUAGES CXX C CUDA)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
set(CUDA_ARCHITECTURES "sm_90" CACHE STRING "List of target GPU architectures" FORCE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    include(${CMAKE_CURRENT_SOURCE_DIR}/configs/linux.cmake)
endif()

# CUDA
set(CUDA_FILES
    src/cuda/CudaUtils.cu
)
include_directories(${CUDA_INCLUDE_DIRS})

# CPP Files
include_directories(src)
set(CPP_FILES
    src/main.cpp
    src/Utils.cpp
    src/clock/Clock.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(${PROJECT_NAME} ${CUDA_FILES} ${CPP_FILES})

set(PCH_HEADER src/pch.h)
target_precompile_headers(${PROJECT_NAME} PRIVATE ${PCH_HEADER})

# CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
