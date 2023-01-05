#!/usr/bin/env python

from conans import ConanFile, CMake, tools

class rocThrustConan(ConanFile):
    name = "rocThrust"
    version = "5.1.0"
    license = "APACHE 2.0"
    author = "ROCm Software Platform Repository"
    # url = "https://nvlabs.github.io/cub/index.html"
    description = "The hipThrust Library"
    topics = ("")
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    def source(self):
        tools.download('https://github.com/ROCmSoftwarePlatform/rocThrust/archive/refs/tags/rocm-5.1.0.tar.gz', 'rocm-5.1.0.tar.gz')
        tools.unzip('rocm-5.1.0.tar.gz')
        return 'rocThrust-rocm-5.1.0'

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder='rocThrust-rocm-5.1.0')
        cmake.build()
        cmake.install()
