#! /usr/bin/env python
# encoding: utf-8

VERSION='0.0.1'
APPNAME='influence-maximization'


def options(opt):
  opt.load('compiler_cxx')
  cfg_options = opt.get_option_group('Configuration options')
  cfg_options.add_option('--spdlog-root', action='store', default='/usr',
                         help='root directory of the installation of spdlog')

  cfg_options.add_option('--trng4-root', action='store', default='/usr',
                         help='root directory of the installation of trng4')

  cfg_options.add_option('--openmp-root', action='store', default='/usr',
                         help='root directory of the installation of openmp')

  cfg_options.add_option('--nlohmann-json-root', action='store', default='/usr',
                         help='root directory of the installation of nlohmann/json')

def configure(conf):
  conf.load('compiler_cxx')

  conf.env.CXXFLAGS += ['-std=c++17',
                        '-O2', '-march=native', '-pipe', '-fomit-frame-pointer']

  conf.check_cxx(
    includes=['{0}/include'.format(conf.options.spdlog_root)],
    header_name='spdlog/spdlog.h',
    uselib_store='SPDLOG',
    msg = "Checking for library spdlog")

  conf.check_cxx(
    includes=['{0}/include'.format(conf.options.nlohmann_json_root)],
    header_name='nlohmann/json.hpp',
    uselib_store='JSON',
    msg = "Checking for library nlohmann/json")

  conf.check_cxx(lib='trng4', uselib_store='TRNG',
                 includes=[ '{0}/include/'.format(conf.options.trng4_root) ],
                 libpath=[ '{0}/lib/'.format(conf.options.trng4_root) ])

  conf.check_cxx(cxxflags=['-fopenmp' ], ldflags=[ '-fopenmp' ],
                 libpath=['{0}/lib/'.format(conf.options.openmp_root)],
                 uselib_store='OpenMP')


def build(bld):
  directories = ['3rd-party', 'include', 'tools']

  bld.recurse(directories)
