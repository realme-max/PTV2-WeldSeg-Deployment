cmake_minimum_required(VERSION 3.25)

project(Ptv2TensorRTRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
    message(FATAL_ERROR "Phase 9A TensorRT runtime currently supports Windows x64 only")
endif()
if(NOT DEFINED TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT must point to the TensorRT 11.1 SDK root")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

find_file(CUDART_RUNTIME_DLL
    NAMES cudart64_12.dll
    HINTS "${CUDAToolkit_ROOT}/bin" "${CUDAToolkit_BIN_DIR}"
    REQUIRED
)

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
set(TENSORRT_BIN_DIR "${TENSORRT_ROOT}/bin")

foreach(required_file
    "${TENSORRT_INCLUDE_DIR}/NvInfer.h"
    "${TENSORRT_INCLUDE_DIR}/NvInferPlugin.h"
    "${TENSORRT_LIB_DIR}/nvinfer_11.lib"
    "${TENSORRT_LIB_DIR}/nvinfer_plugin_11.lib"
    "${TENSORRT_BIN_DIR}/nvinfer_11.dll"
    "${TENSORRT_BIN_DIR}/nvinfer_plugin_11.dll")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required TensorRT SDK file is missing: ${required_file}")
    endif()
endforeach()

add_library(ptv2_tensorrt_runtime STATIC
    src/PluginLoader.cpp
    src/CudaBufferManager.cpp
    src/TensorRTInference.cpp
)

target_include_directories(ptv2_tensorrt_runtime
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${TENSORRT_INCLUDE_DIR}"
)

target_link_libraries(ptv2_tensorrt_runtime
    PUBLIC
        "${TENSORRT_LIB_DIR}/nvinfer_11.lib"
        "${TENSORRT_LIB_DIR}/nvinfer_plugin_11.lib"
        CUDA::cudart
        bcrypt
)

target_compile_definitions(ptv2_tensorrt_runtime PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
    target_compile_options(ptv2_tensorrt_runtime PRIVATE /W4 /WX /utf-8)
endif()

add_executable(trt_runtime_demo src/main.cpp)
target_link_libraries(trt_runtime_demo PRIVATE ptv2_tensorrt_runtime)
if(MSVC)
    target_compile_options(trt_runtime_demo PRIVATE /W4 /WX /utf-8)
endif()

add_custom_command(TARGET trt_runtime_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_BIN_DIR}/nvinfer_11.dll"
        "$<TARGET_FILE_DIR:trt_runtime_demo>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_BIN_DIR}/nvinfer_plugin_11.dll"
        "$<TARGET_FILE_DIR:trt_runtime_demo>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDART_RUNTIME_DLL}"
        "$<TARGET_FILE_DIR:trt_runtime_demo>"
    VERBATIM
)
