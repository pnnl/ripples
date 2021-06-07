#!/usr/bin/env python

from conans import ConanFile, CMake, tools

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
        git = tools.Git(folder="metal")
        git.clone("https://github.com/LLNL/metall.git", "master")
        return "metal"

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="metal")
        cmake.build()
        cmake.install()
