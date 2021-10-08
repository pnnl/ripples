#!/usr/bin/env python

from conans import ConanFile, CMake, tools

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
        tools.download('https://github.com/NVIDIA/cub/archive/1.12.0.tar.gz', 'v1.12.0.tar.gz')
        tools.unzip('v1.12.0.tar.gz')
        return 'cub-1.12.0'

    def build(self):
        cmake = CMake(self)
        cmake.definitions['CMAKE_CUDA_FLAGS'] = '-std=c++14'
        cmake.definitions['CUB_ENABLE_HEADER_TESTING'] = 'OFF'
        cmake.definitions['CUB_ENABLE_TESTING'] = 'OFF'
        cmake.definitions['CUB_ENABLE_THOROUGH_TESTING'] = 'OFF'
        cmake.definitions['CUB_ENABLE_MINIMAL_TESTING'] = 'OFF'
        cmake.definitions['CUB_ENABLE_EXAMPLES'] = 'OFF'
        cmake.set_cmake_flags = True
        cmake.configure(source_folder='cub-1.12.0')
        cmake.build()
        cmake.install()

