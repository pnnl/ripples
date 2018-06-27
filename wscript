#! /usr/bin/env python
# encoding: utf-8

VERSION='0.0.1'
APPNAME='influence-maximization'


def options(opt):
  opt.load('compiler_cxx')

def configure(conf):
  conf.load('compiler_cxx')

  conf.env.CXXFLAGS += ['-std=c++11', '-fopenmp',
                        '-O2', '-march=native', '-pipe', '-fomit-frame-pointer']
  conf.env.LDFLAGS += ['-fopenmp', '-L/usr/local/opt/llvm/lib' ]

  conf.check_cfg(
    package='spdlog', args=['--cflags'], uselib_store='SPDLOG')

  conf.check_cfg(
    package='libtrng', args=['--cflags', '--libs'], uselib_store='TRNG')

def build(bld):
  directories = ['3rd-party', 'include', 'tools']

  bld.recurse(directories)
