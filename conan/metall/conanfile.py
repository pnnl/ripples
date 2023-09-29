#!/usr/bin/env python

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.scm import Git

class MetallConan(ConanFile):
    name = "metall"
    version = "master"
    license = "MIT and Apache 2.0"
    author = "Keita Iwabuchi, Roger Pearce, Maya Gokhale"
    url = "https://github.com/LLNL/metall"
    description = "A Persistent Memory Allocator For Data-Centric Analytics"
    topics = ("Memory Allocation")
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake"

    def requirements(self):
        self.requires("boost/1.75.0")

    def configure(self):
        self.options['boost'].header_only = True

    def source(self):
        git = Git(self)
        git.clone("https://github.com/LLNL/metall.git", target=".")

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
