from conans import ConanFile, AutoToolsBuildEnvironment, tools


class LibtrngConan(ConanFile):
    name = "libtrng"
    version = "4.21"
    license = "<Put the package license here>"
    author = "Heiko Bauke"
    url = "https://www.numbercrunch.de/trng/"
    description = "Tina's Random Number Generator Library"
    topics = ("Pseudo-Random Number Generator")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=True"
    generators = "cmake"

    def source(self):
        tools.download('https://www.numbercrunch.de/trng/trng-4.21.tar.gz', 'trng-4.21.tar.gz')
        tools.unzip('trng-4.21.tar.gz')
        return 'trng-4.12'


    def build(self):
        autotools = AutoToolsBuildEnvironment(self)
        autotools.configure(configure_dir='trng-4.21')
        autotools.make()
        autotools.install()


    def package_info(self):
        self.cpp_info.libs = ["trng4"]

