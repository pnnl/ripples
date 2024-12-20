from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, CMake, cmake_layout


class RipplesConan(ConanFile):
    options = {'gperftools': [True, False],
               'metall' : [True, False],
               'nvidia_cub' : [True, False],
               'enable_benchmarks' : [True, False],
               'gpu' : [None, 'amd', 'nvidia'],
               'metall_checkpointing' : [True, False],
               'weight_type' : ['float', 'uint16', 'uint8'],
               'disable_sorting' : [True, False],
               'speculative_execution' : [True, False],
               'mpi_gather' : [True, False],
               'mpi_chunking' : [True, False],
               'gpu_int_prng' : [True, False]}
    default_options = {'gperftools': True,
                       'nvidia_cub': False,
                       'enable_benchmarks' : False,
                       'metall': False,
                       'gpu' : None,
                       'metall_checkpointing' : True,
                       'weight_type' : 'float',
                       'disable_sorting' : False,
                       'speculative_execution' : True,
                       'mpi_gather' : True,
                       'mpi_chunking' : True,
                       'gpu_int_prng' : False}
    settings = "os", "compiler", "build_type", "arch"

    def configure(self):
        self.options['fmt'].shared = False
        self.options['spdlog'].shared = False
        self.options['metall'].shared = False

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables['RIPPLES_ENABLE_HIP'] = self.options.gpu == 'amd'
        tc.cache_variables['RIPPLES_ENABLE_CUDA'] = self.options.gpu == 'nvidia'
        tc.cache_variables['RIPPLES_ENABLE_METALL'] = self.options.metall
        tc.cache_variables['RIPPLES_ENABLE_BENCHMARKS'] = self.options.enable_benchmarks
        if self.options.metall:
            tc.cache_variables['RIPPLES_ENABLE_METALL_CHECKPOINTING'] = self.options.metall_checkpointing
        else:
            tc.cache_variables['RIPPLES_ENABLE_METALL_CHECKPOINTING'] = False

        if self.options.gperftools:
            tc.cache_variables['RIPPLES_ENABLE_TCMALLOC'] = True
        tc.cache_variables['RIPPLES_ENABLE_FLOAT_WEIGHTS'] = self.options.weight_type == 'float'
        tc.cache_variables['RIPPLES_ENABLE_UINT16_WEIGHTS'] = self.options.weight_type == 'uint16'
        tc.cache_variables['RIPPLES_ENABLE_UINT8_WEIGHTS'] = self.options.weight_type == 'uint8'
        tc.cache_variables['RIPPLES_DISABLE_SORTING'] = self.options.disable_sorting
        tc.cache_variables['RIPPLES_ENABLE_MPI_GATHER'] = self.options.mpi_gather
        if self.options.mpi_gather:
            tc.cache_variables['RIPPLES_ENABLE_SPECULATIVE_EXECUTION'] = self.options.speculative_execution
        else:
            # Speculative execution is not supported without MPI gather
            tc.cache_variables['RIPPLES_ENABLE_SPECULATIVE_EXECUTION'] = False
        tc.cache_variables['RIPPLES_ENABLE_MPI_CHUNKING'] = self.options.mpi_chunking
        tc.cache_variables['RIPPLES_ENABLE_GPU_INT_PRNG'] = self.options.gpu_int_prng
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def requirements(self):
        self.requires('spdlog/1.11.0')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.10')
        self.requires('cli11/2.1.1')
        self.requires('libtrng/4.25')
        if self.options.gperftools:
            self.requires('gperftools/2.15')
        if self.options.enable_benchmarks:
            self.requires('nanobench/4.3.11')
            self.requires('networkit/master')
        if self.options.gpu == 'nvidia' and self.options.nvidia_cub:
            self.requires('nvidia-cub/1.12.0')

        if self.options.gpu == 'amd':
            self.requires('rocthrust/5.3.0')

        if self.options.metall:
            self.requires('metall/0.27')

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        pass

    def package_info(self):
        pass
