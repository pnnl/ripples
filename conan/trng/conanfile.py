#!/usr/bin/env python

from conans import ConanFile, CMake, tools

class LibtrngConan(ConanFile):
    name = "libtrng"
    license = "BSD"
    version = "4.22"
    author = "Heiko Bauke"
    url = "https://www.numbercrunch.de/trng/"
    description = "Tina's Random Number Generator Library"
    topics = ("Pseudo-Random Number Generator")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=True"
    generators = "cmake"
    scm = {
        "type" : "git",
        "url" : "https://github.com/mminutoli/trng4.git",
        "subfolder" : "trng",
        "revision" : "basic_hip_support"
    }

    def build(self):
        cmake = CMake(self)
        cmake.definitions['TRNG_ENABLE_EXAMPLES'] = False
        cmake.definitions['TRNG_ENABLE_TESTS'] = False
        cmake.configure(source_folder='trng')
        cmake.parallel = False
        cmake.build()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["trng4"]
