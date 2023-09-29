#!/usr/bin/env python

import os
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import get, replace_in_file

class LibtrngConan(ConanFile):
    name = "libtrng"
    license = "BSD"
    version = "4.23.1"
    author = "Heiko Bauke"
    description = "Tina's Random Number Generator Library"
    topics = ("Pseudo-Random Number Generator")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared" : True}

    def source(self):
        get(self,
            'https://github.com/rabauke/trng4/archive/refs/tags/v' + self.version + '.tar.gz',
            strip_root=True)
        replace_in_file(self, os.path.join(self.source_folder, "CMakeLists.txt"),
                        "add_subdirectory(examples)", "")
    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["trng4"]
