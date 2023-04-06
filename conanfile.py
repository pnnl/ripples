from conans import ConanFile, tools


class RipplesConan(ConanFile):
    options = {'memkind' : [ True, False],
               'gpu' : [None, 'amd', 'nvidia'],
               'nvidia_cub' : [True, False]}
    default_options = {'memkind' : False,
                       'nvidia_cub' : False,
                       'gpu' : None}
    generators = 'Waf'

    def configure(self):
        self.options['fmt'].shared = False
        self.options['spdlog'].shared = False

    def requirements(self):
        self.requires('spdlog/1.9.2')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.3')
        self.requires('cli11/2.1.1')
        self.requires('libtrng/basic_hip_support@user/stable')
        self.requires('WafGen/0.1@user/stable')
        if self.options.gpu == 'nvidia' and self.options.nvidia_cub:
            self.requires('nvidia-cub/1.12.0@user/stable')

        if self.options.gpu == 'amd':
            self.requires('rocThrust/5.1.0@user/stable')

        if tools.os_info.is_linux and self.options.memkind:
            self.requires('memkind/1.10.1-rc1@memkind/stable')
