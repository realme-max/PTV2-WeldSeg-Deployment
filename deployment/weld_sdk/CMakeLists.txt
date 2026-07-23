cmake_minimum_required(VERSION 3.25)

project(Ptv2WeldSdk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(required_target
    ptv2_pointcloud_pipeline
    ptv2_tensorrt_runtime
    ptv2_postprocess)
    if(NOT TARGET ${required_target})
        message(FATAL_ERROR "${required_target} must be defined before adding weld_sdk")
    endif()
endforeach()

add_library(ptv2_weld_sdk STATIC
    src/WeldConfig.cpp
    src/WeldResult.cpp
    src/WeldDetector.cpp
)

target_include_directories(ptv2_weld_sdk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ptv2_weld_sdk
    PRIVATE
        ptv2_pointcloud_pipeline
        ptv2_tensorrt_runtime
        ptv2_postprocess
)
target_compile_definitions(ptv2_weld_sdk PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

add_executable(sdk_smoke_test tests/sdk_smoke_test.cpp)
target_link_libraries(sdk_smoke_test PRIVATE ptv2_weld_sdk)

add_executable(weld_sdk_testset_qualification tests/weld_sdk_testset_qualification.cpp)
target_link_libraries(weld_sdk_testset_qualification
    PRIVATE
        ptv2_weld_sdk
        ptv2_pointcloud_pipeline
        ptv2_tensorrt_runtime
        Psapi
)

if(MSVC)
    target_compile_options(ptv2_weld_sdk PRIVATE /W4 /WX /utf-8)
    target_compile_options(sdk_smoke_test PRIVATE /W4 /WX /utf-8)
    target_compile_options(weld_sdk_testset_qualification PRIVATE /W4 /WX /utf-8)
endif()

if(DEFINED TENSORRT_ROOT AND DEFINED CUDART_RUNTIME_DLL)
    foreach(runtime_executable sdk_smoke_test weld_sdk_testset_qualification)
        add_custom_command(TARGET ${runtime_executable} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TENSORRT_ROOT}/bin/nvinfer_11.dll"
                "$<TARGET_FILE_DIR:${runtime_executable}>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TENSORRT_ROOT}/bin/nvinfer_plugin_11.dll"
                "$<TARGET_FILE_DIR:${runtime_executable}>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDART_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${runtime_executable}>"
            VERBATIM
        )
    endforeach()
endif()
