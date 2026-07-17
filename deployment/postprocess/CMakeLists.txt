cmake_minimum_required(VERSION 3.25)

project(Ptv2Postprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET ptv2_pointcloud_pipeline)
    message(FATAL_ERROR "ptv2_pointcloud_pipeline must be defined before adding postprocess")
endif()

add_library(ptv2_postprocess STATIC
    src/SegmentationPostProcessor.cpp
    src/CoordinateRecovery.cpp
    src/WeldGeometryExtractor.cpp
    src/ResultWriter.cpp
)

target_include_directories(ptv2_postprocess PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ptv2_postprocess PUBLIC ptv2_pointcloud_pipeline)
target_compile_definitions(ptv2_postprocess PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

add_executable(postprocess_failure_probe tests/PostprocessFailureProbe.cpp)
target_link_libraries(postprocess_failure_probe PRIVATE ptv2_postprocess)

if(MSVC)
    target_compile_options(ptv2_postprocess PRIVATE /W4 /WX /utf-8)
    target_compile_options(postprocess_failure_probe PRIVATE /W4 /WX /utf-8)
endif()
