cmake_minimum_required(VERSION 3.25)

project(Ptv2PointCloudPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ptv2_pointcloud_pipeline STATIC
    src/PointCloudLoader.cpp
    src/PointSampler.cpp
    src/FeatureBuilder.cpp
    src/KnnGraphBuilder.cpp
)

target_include_directories(ptv2_pointcloud_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(ptv2_pointcloud_pipeline PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
    target_compile_options(ptv2_pointcloud_pipeline PRIVATE /W4 /WX /utf-8)
endif()
