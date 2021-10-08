from conans import ConanFile, tools


class RipplesConan(ConanFile):
    options = {'memkind' : [ True, False],
               'nvidia_cub' : [True, False]}
    default_options = {'memkind' : False,
                       'nvidia_cub' : False}
    generators = 'Waf'

    def configure(self):
        self.options['fmt'].shared = True
        self.options['spdlog'].shared = True

    def requirements(self):
        self.requires('spdlog/1.9.2')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.3')
        self.requires('cli11/2.1.1')
        self.requires('libtrng/4.22@user/stable')
        self.requires('WafGen/0.1@user/stable')
        if self.options.nvidia_cub:
            self.requires('nvidia-cub/1.12.0@user/stable')

        if tools.os_info.is_linux and self.options.memkind:
            self.requires('memkind/1.10.1-rc1@memkind/stable')
