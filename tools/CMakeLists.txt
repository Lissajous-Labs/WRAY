add_executable(wray_dump dump/dump.cpp)
target_link_libraries(wray_dump PRIVATE wray::wray wray_warnings)
