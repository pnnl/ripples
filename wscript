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

  if conf.env.COMPILER_CXX == 'g++':
    conf.env.LDFLAGS += ['-fopenmp']
  else:
    # conf.env.LDFLAGS += ['-L/usr/local/opt/llvm/lib', '-lomp']
    conf.env.LDFLAGS += ['-L/usr/local/opt/llvm/lib', '-fopenmp=libomp']

  conf.load('boost')
  # Using boost for command line arguments
  conf.check_boost('program_options')

  conf.check_cfg(
    package='RapidJSON', args=['--cflags'], uselib_store='RAPIDJSON')

def build(bld):
  directories = ['include', 'tools']

  bld.recurse(directories)
