#!/usr/bin/env python
# encoding: utf-8

"""Tool to detect spdlog."""


def options(opt):
    opt_group = opt.add_option_group('Configuration options')
    opt_group.add_option(
        '--trng4-root', action='store', default='/usr',
        help='root directory of the installation of trng4')

def configure(conf):
    conf.check_cxx(lib='trng4', uselib_store='TRNG',
                   includes=['{0}/include/'.format(conf.options.trng4_root)],
                   libpath=['{0}/lib/'.format(conf.options.trng4_root)])
