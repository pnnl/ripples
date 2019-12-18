from conans import ConanFile, tools


class RipplesConan(ConanFile):
    options = {'memkind' : [ True, False]}
    default_options = {'memkind' : False}
    generators = 'Waf'

    def requirements(self):
        self.requires('spdlog/1.3.1@bincrafters/stable')
        self.requires('jsonformoderncpp/3.7.0@vthiery/stable')
        self.requires('Catch2/2.9.2@catchorg/stable')
        self.requires('CLI11/1.8.0@cliutils/stable')
        self.requires('libtrng/4.21@user/stable')
        self.requires('WafGen/0.1@user/stable')

        if tools.os_info.is_linux and self.options.memkind:
            self.requires('memkind/1.9.0@memkind/stable')
