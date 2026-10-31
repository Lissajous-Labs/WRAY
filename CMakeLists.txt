cmake_minimum_required(VERSION 3.24)
project(wray VERSION 0.1.0 LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(WRAY_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(WRAY_BUILD_BENCH "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(WRAY_BUILD_TOOLS "Build tools" ${PROJECT_IS_TOP_LEVEL})
set(WRAY_SIMD "auto" CACHE STRING "auto or scalar")
set_property(CACHE WRAY_SIMD PROPERTY STRINGS auto scalar)

add_library(wray INTERFACE)
add_library(wray::wray ALIAS wray)
target_include_directories(wray INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_features(wray INTERFACE cxx_std_23)
if(WRAY_SIMD STREQUAL "scalar")
    target_compile_definitions(wray INTERFACE WRAY_SIMD_SCALAR=1)
endif()

add_library(wray_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(wray_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
endif()

if(WRAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WRAY_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(WRAY_BUILD_BENCH)
    add_subdirectory(bench)
endif()
