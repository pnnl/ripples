from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, CMake, cmake_layout


class RipplesConan(ConanFile):
    options = {'metall' : [True, False],
               'nvidia_cub' : [True, False],
               'gpu' : [None, 'amd', 'nvidia']}
    default_options = {'nvidia_cub' : False,
                       'metall': False,
                       'gpu' : None}
    settings = "os", "compiler", "build_type", "arch"

    def configure(self):
        self.options['fmt'].shared = False
        self.options['spdlog'].shared = False

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables['RIPPLES_ENABLE_HIP'] = self.options.gpu == 'amd'
        tc.cache_variables['RIPPLES_ENABLE_CUDA'] = self.options.gpu == 'nvidia'
        tc.cache_variables['RIPPLES_ENABLE_METALL'] = self.options.metall
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def requirements(self):
        self.requires('spdlog/1.11.0')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.10')
        self.requires('cli11/2.1.1')
        self.requires('libtrng/4.23.1')
        if self.options.gpu == 'nvidia' and self.options.nvidia_cub:
            self.requires('nvidia-cub/1.12.0')

        if self.options.gpu == 'amd':
            self.requires('rocthrust/5.3.0')

        if self.options.metall:
            self.requires('metall/master')

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        pass

    def package_info(self):
        pass
