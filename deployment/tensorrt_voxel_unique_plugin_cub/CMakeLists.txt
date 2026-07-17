cmake_minimum_required(VERSION 3.25)

project(Ptv2VoxelUniqueCubPluginLibrary LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT must point to the TensorRT SDK root")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

add_library(ptv2_voxel_unique_cub_plugin SHARED
    VoxelUniqueCubPlugin.cpp
    VoxelUniqueCubPlugin.cu
    VoxelUniqueCubPluginLibrary.cpp
)

target_include_directories(ptv2_voxel_unique_cub_plugin PRIVATE
    "${TENSORRT_ROOT}/include"
)

target_link_libraries(ptv2_voxel_unique_cub_plugin PRIVATE
    "${TENSORRT_ROOT}/lib/nvinfer_11.lib"
    CUDA::cudart
)

set_property(TARGET ptv2_voxel_unique_cub_plugin PROPERTY CUDA_ARCHITECTURES 120)

target_compile_definitions(ptv2_voxel_unique_cub_plugin PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

if(MSVC)
    target_compile_options(ptv2_voxel_unique_cub_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
endif()
