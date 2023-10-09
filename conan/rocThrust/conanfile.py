#!/usr/bin/env python

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import get


class rocThrustConan(ConanFile):
    name = "rocthrust"
    version = "5.3.0"
    license = "APACHE 2.0"
    author = "ROCm Software Platform Repository"
    # url = "https://nvlabs.github.io/cub/index.html"
    description = "The hipThrust Library"
    topics = ("")
    settings = "os", "compiler", "build_type", "arch"

    def source(self):
        get(self, 'https://github.com/ROCmSoftwarePlatform/rocThrust/archive/refs/tags/rocm-5.3.0.tar.gz',
            strip_root=True)

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
