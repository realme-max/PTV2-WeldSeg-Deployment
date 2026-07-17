cmake_minimum_required(VERSION 3.25)

project(Ptv2VoxelUniqueCubCorrectness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT must point to the TensorRT SDK root")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

add_executable(voxel_unique_cub_correctness correctness_main.cpp)
target_include_directories(voxel_unique_cub_correctness PRIVATE "${TENSORRT_ROOT}/include")
target_link_libraries(voxel_unique_cub_correctness PRIVATE
    "${TENSORRT_ROOT}/lib/nvinfer_11.lib"
    "${TENSORRT_ROOT}/lib/nvinfer_plugin_11.lib"
    "${TENSORRT_ROOT}/lib/nvonnxparser_11.lib"
    CUDA::cudart
)
target_compile_definitions(voxel_unique_cub_correctness PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
    target_compile_options(voxel_unique_cub_correctness PRIVATE /W4 /utf-8)
endif()
