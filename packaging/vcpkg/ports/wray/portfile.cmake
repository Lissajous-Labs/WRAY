set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Lissajous-Labs/WRAY
    REF "v${VERSION}"
    SHA512 e0173c107359b67a4c411d858cd5243781149ebf53332eb26a5536cbf5f87c7b461f16f19075b08c72c8f0dcb91c8a83d86c9c6bd844c5e1821781a56d37052d
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DWRAY_BUILD_TESTS=OFF
        -DWRAY_BUILD_TOOLS=OFF
        -DWRAY_BUILD_BENCH=OFF
        -DWRAY_INSTALL=ON
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/wray)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
