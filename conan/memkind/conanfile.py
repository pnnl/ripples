from conans import ConanFile, AutoToolsBuildEnvironment, tools


class MemkindConan(ConanFile):
    name = "memkind"
    version = "1.9.0"
    license = "BSD-3"
    author = "<Put your name here> <And your email here>"
    url = "https://github.com/memkind/memkind/releases/tag/v1.9.0"
    description = "<Description of Memkind here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=False"
    generators = "cmake"

    def source(self):
        tools.download('https://github.com/memkind/memkind/archive/v1.9.0.tar.gz', 'memkind-1.9.0.tar.gz')
        tools.unzip('memkind-1.9.0.tar.gz')
        return 'memkind-1.9.0'

    def build(self):
        with tools.chdir('memkind-1.9.0'):
            self.run("./build_jemalloc.sh")
            self.run("./autogen.sh")
            autotools = AutoToolsBuildEnvironment(self)
            autotools.configure()
            autotools.make()
            autotools.install()


    def package_info(self):
        self.cpp_info.libs = ["memkind"]
