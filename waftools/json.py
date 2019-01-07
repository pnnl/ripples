#!/usr/bin/env python
# encoding: utf-8

"""Tool to detect json library."""


def options(opt):
    opt_group = opt.add_option_group('Configuration options')
    opt.add_option(
        '--nlohmann-json-root', action='store', default='/usr',
        help='root directory of the installation of nlohmann/json')


def configure(conf):
    conf.check_cxx(
        includes=['{0}/include'.format(conf.options.nlohmann_json_root)],
        header_name='nlohmann/json.hpp',
        uselib_store='JSON',
        msg='Checking for library nlohmann/json')
