#!/usr/bin/env python
# encoding: utf-8

"""Tool to detect spdlog."""


def options(opt):
    opt_group = opt.add_option_group('Configuration options')
    opt_group.add_option(
        '--spdlog-root', action='store', default='/usr',
        help='root directory of the installation of spdlog')


def configure(conf):
    conf.check_cxx(
        includes=['{0}/include'.format(conf.options.spdlog_root)],
        header_name='spdlog/spdlog.h',
        uselib_store='SPDLOG',
        msg='Checking for library spdlog')
