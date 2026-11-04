cmake_minimum_required(VERSION 3.24)
project(wray_juce_consumer VERSION 1.0 LANGUAGES C CXX)

set(JUCE_SOURCE_DIR "" CACHE PATH "JUCE 8 source tree")
if(NOT EXISTS "${JUCE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set JUCE_SOURCE_DIR to a JUCE 8 source tree")
endif()
add_subdirectory("${JUCE_SOURCE_DIR}" juce)

juce_add_module("${CMAKE_CURRENT_SOURCE_DIR}/../../../include/wray")

juce_add_console_app(wray_juce_consumer PRODUCT_NAME "WRAY JUCE consumer")
target_sources(wray_juce_consumer PRIVATE main.cpp)
target_compile_definitions(wray_juce_consumer PRIVATE JUCE_USE_CURL=0 JUCE_WEB_BROWSER=0)
target_compile_options(wray_juce_consumer PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffp-contract=off>)
target_link_libraries(wray_juce_consumer PRIVATE wray juce::juce_core juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
