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

  conf.env.LDFLAGS += ['-fopenmp']

  conf.load('boost')
  # Using boost for command line arguments
  conf.check_boost('program_options')

  conf.check_cfg(
    package='RapidJSON', args=['--cflags'], uselib_store='RAPIDJSON')

  conf.check_cfg(
    package='benchmark', args=['--cflags', '--libs'], uselib_store='BENCHMARK')

def build(bld):
  directories = ['3rd-party', 'include', 'tools']

  bld.recurse(directories)
