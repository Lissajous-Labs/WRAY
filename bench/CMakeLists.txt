add_executable(wray_bench bench.cpp)
target_link_libraries(wray_bench PRIVATE wray::wray)
string(TOUPPER "${CMAKE_BUILD_TYPE}" wray_bench_config)
target_compile_definitions(wray_bench PRIVATE WRAY_BENCH_FLAGS="${CMAKE_CXX_COMPILER_ID} ${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${wray_bench_config}} WRAY_SIMD=${WRAY_SIMD}")
