#! /usr/bin/env python
# encoding: utf-8

VERSION='0.0.1'
APPNAME='influence-maximization'


def options(opt):
  opt.load('compiler_cxx')

def configure(conf):
  conf.load('compiler_cxx')

  conf.env.CXXFLAGS += ['-std=c++17',
                        '-O2', '-march=native', '-pipe', '-fomit-frame-pointer']

  conf.check_cfg(
    package='spdlog', args=['--cflags'], uselib_store='SPDLOG')

  conf.check_cxx(lib='trng4', uselib_store='TRNG')

  conf.check_cxx(cxxflags=['-fopenmp' ], ldflags=[ '-fopenmp' ], libpath=['/usr/local/opt/llvm/lib/'], uselib_store='OpenMP')


def build(bld):
  directories = ['3rd-party', 'include', 'tools']

  bld.recurse(directories)
