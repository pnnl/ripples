#!/usr/bin/env python

from conans import ConanFile, CMake, tools

class LibtrngConan(ConanFile):
    name = "libtrng"
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
        tools.download('https://github.com/rabauke/trng4/archive/refs/tags/v' + self.version + '.tar.gz', 'trng-' + self.version + '.tar.gz')
        tools.unzip('trng-' + self.version + '.tar.gz')
        return 'trng4-' + self.version

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder='trng4-' + self.version)
        cmake.parallel = False
        cmake.build()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["trng4"]
