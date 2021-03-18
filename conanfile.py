from conans import ConanFile, tools


class RipplesConan(ConanFile):
    options = {'memkind' : [ True, False]}
    default_options = {'memkind' : False}
    generators = 'Waf'

    def requirements(self):
        self.requires('spdlog/1.3.1@bincrafters/stable')
        self.requires('nlohmann_json/3.9.1')
        self.requires('catch2/2.13.3')
        self.requires('CLI11/1.8.0@cliutils/stable')
        self.requires('libtrng/4.22@user/stable')
        self.requires('WafGen/0.1@user/stable')
        self.requires('nvidia-cub/1.12.0@user/stable')

        if tools.os_info.is_linux and self.options.memkind:
            self.requires('memkind/1.10.1-rc1@memkind/stable')
