cmake_minimum_required(VERSION 3.24)
project(wray_fetch_content_consumer LANGUAGES CXX)

set(WRAY_GIT_REPOSITORY "https://github.com/Lissajous-Labs/WRAY.git" CACHE STRING "Repository to fetch WRAY from")
set(WRAY_GIT_TAG "v1.0.5" CACHE STRING "Release tag or commit to fetch")

include(FetchContent)
FetchContent_Declare(wray GIT_REPOSITORY "${WRAY_GIT_REPOSITORY}" GIT_TAG "${WRAY_GIT_TAG}")
FetchContent_MakeAvailable(wray)

foreach(option IN ITEMS WRAY_BUILD_TESTS WRAY_BUILD_TOOLS WRAY_BUILD_BENCH WRAY_INSTALL)
    if(${option})
        message(FATAL_ERROR "${option} defaults to ON in a fetched WRAY")
    endif()
endforeach()

add_executable(wray_fetch_content_consumer ../package/main.cpp)
target_link_libraries(wray_fetch_content_consumer PRIVATE wray::wray)
