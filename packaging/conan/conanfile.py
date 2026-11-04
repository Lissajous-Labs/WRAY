import os

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get

required_conan_version = ">=2.0"


class WrayConan(ConanFile):
    name = "wray"
    description = "Verified SIMD math for audio DSP"
    license = "GPL-2.0-or-later"
    url = "https://github.com/Lissajous-Labs/WRAY"
    homepage = "https://github.com/Lissajous-Labs/WRAY"
    topics = ("audio", "dsp", "simd", "math", "header-only")
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True

    def _released(self):
        return str(self.version) in self.conan_data["sources"]

    def export_sources(self):
        if self._released():
            return
        root = os.path.join(self.recipe_folder, "..", "..")
        for pattern in ("CMakeLists.txt", "LICENSE", "cmake/*", "include/*"):
            copy(self, pattern, root, self.export_sources_folder)

    def layout(self):
        cmake_layout(self)

    def package_id(self):
        self.info.clear()

    def validate(self):
        check_min_cppstd(self, 20)

    def source(self):
        if self._released():
            get(self, **self.conan_data["sources"][str(self.version)], strip_root=True)

    def generate(self):
        toolchain = CMakeToolchain(self)
        for option in ("WRAY_BUILD_TESTS", "WRAY_BUILD_TOOLS", "WRAY_BUILD_BENCH"):
            toolchain.cache_variables[option] = False
        toolchain.cache_variables["WRAY_INSTALL"] = True
        toolchain.generate()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.configure()
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = [os.path.join("lib", "cmake", "wray")]
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.cxxflags = ["/fp:precise"] if self.settings.compiler == "msvc" else ["-ffp-contract=off"]
