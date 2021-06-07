#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2019, Battelle Memorial Institute
#
# Battelle Memorial Institute (hereinafter Battelle) hereby grants permission to
# any person or entity lawfully obtaining a copy of this software and associated
# documentation files (hereinafter “the Software”) to redistribute and use the
# Software in source and binary forms, with or without modification.  Such
# person or entity may use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and may permit others to do
# so, subject to the following conditions:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimers.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Other than as used herein, neither the name Battelle Memorial Institute or
#    Battelle may be used in any form whatsoever without the express written
#    consent of Battelle.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

VERSION = '0.0.1'
APPNAME = 'ripples'


def options(opt):
    opt.load('compiler_cxx')
    opt.load('waf_unit_test')
    opt.load('sphinx', tooldir='waftools')

    cfg_options = opt.get_option_group('Configuration options')

    opt.load('trng4', tooldir='waftools')
    opt.load('libjson', tooldir='waftools')
    opt.load('spdlog', tooldir='waftools')

    cfg_options.add_option(
        '--openmp-root', action='store', default='/usr',
        help='root directory of the installation of openmp')

    cfg_options.add_option(
        '--enable-mpi', action='store_true', default=False,
        help='enable openmpi implementation')

    cfg_options.add_option(
        '--enable-cuda', action='store_true', default=False,
        help='enable cuda implementation')

    opt.load('mpi', tooldir='waftools')
    opt.load('cuda', tooldir='waftools')
    opt.load('memkind', tooldir='waftools')
    opt.load('metall', tooldir='waftools')


def configure(conf):
    try:
        build_dir = conf.options.out if conf.options.out != '' else 'build'
        conf.load('waf_conan_libs_info', tooldir=[build_dir, '.'])
    except:
        pass

    conf.load('compiler_cxx')
    conf.load('clang_compilation_database')
    conf.load('waf_unit_test')
    conf.load('sphinx', tooldir='waftools')

    if conf.options.enable_metall:
        conf.env.CXXFLAGS += ['-std=c++17', '-pipe']
    else:
        conf.env.CXXFLAGS += ['-std=c++14', '-pipe']

    conf.load('spdlog', tooldir='waftools')
    conf.load('libjson', tooldir='waftools')
    conf.load('trng4', tooldir='waftools')
    conf.load('catch2', tooldir='waftools')
    conf.load('cli', tooldir='waftools')

    conf.check_cxx(cxxflags=['-fopenmp'], ldflags=['-fopenmp'],
                   libpath=['{0}'.format(conf.options.openmp_root)],
                   uselib_store='OpenMP')

    if conf.options.enable_mpi:
        conf.load('mpi', tooldir='waftools')

    conf.env.ENABLE_CUDA=False
    if conf.options.enable_cuda:
        conf.load('cuda', tooldir='waftools')
        conf.env.ENABLE_CUDA = True
        conf.env.CUDAFLAGS = ['--expt-relaxed-constexpr']

    if conf.options.enable_memkind and conf.options.enable_metall:
        conf.error('Metall and Memkind are mutually exclusive')

    conf.env.ENABLE_MEMKIND=False
    if conf.options.enable_memkind:
        conf.load('memkind', tooldir='waftools')
        conf.env.ENABLE_MEMKIND=True

    conf.env.ENABLE_METALL=False
    if conf.options.enable_metall:
        conf.load('metall', tooldir='waftools')
        conf.env.ENABLE_METALL=True

    env = conf.env
    conf.setenv('release', env)
    conf.env.append_value('CXXFLAGS', ['-O3', '-mtune=native'])

    conf.setenv('debug', env)
    conf.env.append_value('CXXFLAGS', ['-g', '-DDEBUG'])
    if conf.env.CXX == 'clang++':
        conf.env.append_value('CXXFLAGS', ['-O1', '-fsanitize=address', '-fno-omit-frame-pointer'])
    conf.env.append_value('CUDAFLAGS', ['-DDEBUG'])


def build(bld):
    if not bld.variant:
        bld.fatal('call "./waf build_release" or "./waf build_debug", and try "./waf --help"')
    directories = ['include', 'tools', 'test']

    bld.recurse(directories)

    from waflib.Tools import waf_unit_test
    bld.add_post_fun(waf_unit_test.summary)


def build_docs(bld):
    if bld.env.ENABLE_DOCS:
        bld(features='sphinx', sources='docs')
    else:
        bld.fatal('Please configure with --enable-docs')


from waflib import Build
class docs(Build.BuildContext):
    fun = 'build_docs'
    cmd = 'docs'


from waflib.Build import BuildContext, CleanContext, InstallContext, UninstallContext
for x in 'debug release'.split():
    for y in (BuildContext, CleanContext, InstallContext, UninstallContext):
        name = y.__name__.replace('Context', '').lower()
        class tmp(y):
            cmd = name + '_' + x
            variant = x
