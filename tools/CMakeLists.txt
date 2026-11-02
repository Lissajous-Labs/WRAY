add_executable(wray_dump dump/dump.cpp)
target_link_libraries(wray_dump PRIVATE wray::wray wray_warnings)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    add_executable(wray_dense dense/dense.cpp)
    target_link_libraries(wray_dense PRIVATE wray::wray quadmath)
    set_target_properties(wray_dense PROPERTIES CXX_EXTENSIONS ON)
endif()
