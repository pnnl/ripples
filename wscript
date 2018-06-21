#! /usr/bin/env python
# encoding: utf-8

VERSION='0.0.1'
APPNAME='influence-maximization'


def options(opt):
  opt.load('compiler_cxx')

def configure(conf):
  conf.load('compiler_cxx')

  conf.env.CXXFLAGS += ['-std=c++11', '-fopenmp']
  conf.env.CXXFLAGS += ['-Ofast', '-march=native']
  conf.env.LDFLAGS += ['-L/usr/local/opt/llvm/lib', '-liomp5']

  conf.check_cfg(
    package='spdlog', args=['--cflags'], uselib_store='SPDLOG')

  conf.check_cfg(
    package='libtrng', args=['--cflags', '--libs'], uselib_store='TRNG')

def build(bld):
  directories = ['3rd-party', 'include', 'tools']

  bld.recurse(directories)
