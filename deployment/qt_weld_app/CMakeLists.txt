cmake_minimum_required(VERSION 3.25)

project(Ptv2WeldQtSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if(NOT WIN32)
    message(FATAL_ERROR "Phase 10A Qt application currently supports Windows x64 only")
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

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Test)
if(NOT DEFINED QT_VERSION_MAJOR)
    set(QT_VERSION_MAJOR 5)
endif()
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Test)

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

add_library(ptv2_weld_qt_frontend STATIC
    include/MainWindow.h
    include/WeldDetectionWorker.h
    include/QtWeldResultViewModel.h
    src/MainWindow.cpp
    src/WeldDetectionWorker.cpp
    src/QtWeldResultViewModel.cpp
)
target_include_directories(ptv2_weld_qt_frontend PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ptv2_weld_qt_frontend
    PUBLIC
        ptv2_weld_sdk
        Qt${QT_VERSION_MAJOR}::Widgets
)
target_compile_definitions(ptv2_weld_qt_frontend PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

add_executable(ptv2_weld_qt_smoke src/main.cpp)
target_link_libraries(ptv2_weld_qt_smoke
    PRIVATE
        ptv2_weld_qt_frontend
        ptv2_weld_sdk
        Qt${QT_VERSION_MAJOR}::Widgets
)

add_executable(QtSdkIntegrationSmoke tests/QtSdkIntegrationSmoke.cpp)
target_link_libraries(QtSdkIntegrationSmoke
    PRIVATE
        ptv2_weld_qt_frontend
        ptv2_weld_sdk
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Test
)

if(MSVC)
    target_compile_options(ptv2_weld_qt_frontend PRIVATE /W4 /WX /utf-8)
    target_compile_options(ptv2_weld_qt_smoke PRIVATE /W4 /WX /utf-8)
    target_compile_options(QtSdkIntegrationSmoke PRIVATE /W4 /WX /utf-8)
endif()

foreach(runtime_target ptv2_weld_qt_smoke QtSdkIntegrationSmoke)
    add_custom_command(TARGET ${runtime_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_ROOT}/bin/nvinfer_11.dll"
            "$<TARGET_FILE_DIR:${runtime_target}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_ROOT}/bin/nvinfer_plugin_11.dll"
            "$<TARGET_FILE_DIR:${runtime_target}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDART_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:${runtime_target}>"
        VERBATIM
    )
endforeach()

enable_testing()
if(DEFINED PTV2_TEST_ENGINE
    AND DEFINED PTV2_TEST_PLUGIN
    AND DEFINED PTV2_TEST_ENGINE_SHA256
    AND DEFINED PTV2_TEST_CLOUD
    AND DEFINED PTV2_TEST_SMALL_CLOUD
    AND DEFINED PTV2_TEST_ARTIFACT_DIR)
    add_test(
        NAME QtSdkIntegrationSmoke
        COMMAND QtSdkIntegrationSmoke
            --engine "${PTV2_TEST_ENGINE}"
            --plugin "${PTV2_TEST_PLUGIN}"
            --engine-sha256 "${PTV2_TEST_ENGINE_SHA256}"
            --cloud "${PTV2_TEST_CLOUD}"
            --small-cloud "${PTV2_TEST_SMALL_CLOUD}"
            --artifact-dir "${PTV2_TEST_ARTIFACT_DIR}"
    )
endif()
