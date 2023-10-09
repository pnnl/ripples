#!/usr/bin/env python

import os
from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.scm import Git

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
        git = Git(self)
        clone_args = ['--depth', '1', '--branch', 'basic_hip_support']
        git.clone(url='https://github.com/mminutoli/trng4.git',
                  args=clone_args, target='.')

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables['TRNG_ENABLE_EXAMPLES'] = False
        tc.cache_variables['TRNG_ENABLE_TESTS'] = False
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
