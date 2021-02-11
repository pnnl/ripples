#!/usr/bin/env python

from conans import ConanFile, CMake, tools

class LibtrngConan(ConanFile):
    name = "libtrng"
    version = "4.22"
    license = "BSD"
    author = "Heiko Bauke"
    url = "https://www.numbercrunch.de/trng/"
    description = "Tina's Random Number Generator Library"
    topics = ("Pseudo-Random Number Generator")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=True"
    generators = "cmake"

    def source(self):
        tools.download('https://www.numbercrunch.de/trng/trng-4.22.tar.gz', 'trng-4.22.tar.gz')
        tools.unzip('trng-4.22.tar.gz')
        return 'trng4-4.22'

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder='trng4-4.22')
        cmake.parallel = False
        cmake.build()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["trng4"]
