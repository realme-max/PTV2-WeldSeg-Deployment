cmake_minimum_required(VERSION 3.25)

project(Ptv2WeldTensorRTApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
    message(FATAL_ERROR "Phase 9B weld_trt_app currently supports Windows x64 only")
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

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../pointcloud_pipeline"
    "${CMAKE_CURRENT_BINARY_DIR}/pointcloud_pipeline"
)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../postprocess"
    "${CMAKE_CURRENT_BINARY_DIR}/postprocess"
)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_runtime"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorrt_runtime"
)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../weld_sdk"
    "${CMAKE_CURRENT_BINARY_DIR}/weld_sdk"
)

add_executable(weld_trt_demo src/main.cpp)
target_link_libraries(weld_trt_demo PRIVATE ptv2_weld_sdk)
if(MSVC)
    target_compile_options(weld_trt_demo PRIVATE /W4 /WX /utf-8)
endif()

add_custom_command(TARGET weld_trt_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/bin/nvinfer_11.dll"
        "$<TARGET_FILE_DIR:weld_trt_demo>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/bin/nvinfer_plugin_11.dll"
        "$<TARGET_FILE_DIR:weld_trt_demo>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDART_RUNTIME_DLL}"
        "$<TARGET_FILE_DIR:weld_trt_demo>"
    VERBATIM
)
