foreach(name simd numerics special parity accuracy)
    add_executable(wray_${name}_test ${name}_test.cpp)
    target_link_libraries(wray_${name}_test PRIVATE wray::wray wray_warnings)
    add_test(NAME ${name} COMMAND wray_${name}_test)
endforeach()

set_tests_properties(accuracy PROPERTIES LABELS "accuracy")
add_test(NAME accuracy_exhaustive COMMAND wray_accuracy_test --exhaustive)
set_tests_properties(accuracy_exhaustive PROPERTIES LABELS "exhaustive")

set(WRAY_PYTHON "${PROJECT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python with mpmath and numpy")
if(EXISTS "${WRAY_PYTHON}" AND WRAY_BUILD_TOOLS)
    add_test(NAME oracle COMMAND "${WRAY_PYTHON}" "${PROJECT_SOURCE_DIR}/tools/oracle/oracle.py"
             --dump $<TARGET_FILE:wray_dump> --out "${CMAKE_CURRENT_BINARY_DIR}/accuracy.json" --check)
    set_tests_properties(oracle PROPERTIES LABELS "oracle")
endif()
