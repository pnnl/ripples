#!/usr/bin/env python

from conan import ConanFile
from conan.tools.gnu import AutotoolsToolchain, Autotools
from conan.tools.files import get, replace_in_file


class MemkindConan(ConanFile):
    name = "memkind"
    version = "1.10.1-rc1"
    license = "BSD-3"
    author = "<Put your name here> <And your email here>"
    url = "https://github.com/memkind/memkind/releases/tag/v1.10.1-rc1"
    description = "<Description of Memkind here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}

    def source(self):
        get(self, 'https://github.com/memkind/memkind/archive/v1.10.1-rc1.tar.gz',
            strip_root=True)
        replace_in_file(self, 'Makefile.am', 'include examples/Makefile.mk', '')

    def generate(self):
        autotools = AutotoolsToolchain(self)
        autotools.generate()

    def build(self):
        autotools = Autotools(self)
        autotools.autoreconf()
        autotools.configure()
        autotools.make()

    def package(self):
        autotools = Autotools(self)
        autotools.install()


    def package_info(self):
        self.cpp_info.libs = ["memkind"]
