from conans import ConanFile, tools


class RipplesConan(ConanFile):
    options = {'memkind' : [ True, False],
               'metal' : [True, False],
               'nvidia_cub' : [True, False],
               'gpu' : [None, 'amd', 'nvidia']}
    default_options = {'memkind' : False,
                       'nvidia_cub' : False,
                       'metal': False,
                       'gpu' : None}
    generators = 'Waf'
    settings = "os", "compiler", "build_type", "arch"

    def configure(self):
        self.options['fmt'].shared = False
        self.options['spdlog'].shared = False

    def requirements(self):
        self.requires('spdlog/1.11.0')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.3')
        self.requires('cli11/2.1.1')
        self.requires('libtrng/4.22@user/stable')
        self.requires('WafGen/0.1@user/stable')
        if self.options.gpu == 'nvidia' and self.options.nvidia_cub:
            self.requires('nvidia-cub/1.12.0@user/stable')

        if self.options.gpu == 'amd':
            self.requires('rocThrust/5.1.0@user/stable')

        if self.options.memkind and self.options.metal:
            self.output.error("Metal and Memkind are mutually exclusive")

        if tools.os_info.is_linux:
            if self.options.memkind:
                self.requires('memkind/1.10.1-rc1@memkind/stable')

        if self.options.metal:
            self.requires('metall/master@user/stable')
