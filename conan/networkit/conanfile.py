#!/usr/bin/env python

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.scm import Git

class MetallConan(ConanFile):
    name = "networkit"
    version = "master"
    license = "MIT and Apache 2.0"
    author = ""
    url = "https://github.com/networkit/networkit.git"
    description = ""
    topics = ("Network Analysis")
    settings = "os", "compiler", "build_type", "arch"

    def source(self):
        git = Git(self)
        git.clone(self.url, target=".")
        git.run("submodule update --init")

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
        self.cpp_info.libs = ["networkit"]
