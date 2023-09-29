#!/usr/bin/env python

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.files import get

class NvidaCubConan(ConanFile):
    name = "nvidia-cub"
    version = "1.12.0"
    license = "BSD"
    author = "NVIDA"
    url = "https://nvlabs.github.io/cub/index.html"
    description = "The CUB Library"
    topics = ("Reusable components for the CUDA programming model.")
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    def source(self):
        get(self, 'https://github.com/NVIDIA/cub/archive/1.12.0.tar.gz',
            strip_root=True)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.variables['CMAKE_CUDA_FLAGS'] = '-std=c++14'
        cmake.variables['CUB_ENABLE_HEADER_TESTING'] = False
        cmake.variables['CUB_ENABLE_TESTING'] = False
        cmake.variables['CUB_ENABLE_THOROUGH_TESTING'] = False
        cmake.variables['CUB_ENABLE_MINIMAL_TESTING'] = False
        cmake.variables['CUB_ENABLE_EXAMPLES'] = False
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
