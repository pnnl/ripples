#! /usr/bin/env python
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

    opt.load('mpi', tooldir='waftools')


def configure(conf):
    conf.load('compiler_cxx')
    conf.load('clang_compilation_database')
    conf.load('waf_unit_test')
    conf.load('sphinx', tooldir='waftools')

    conf.env.CXXFLAGS += ['-std=c++17', '-O3', '-march=native', '-pipe']

    conf.load('spdlog', tooldir='waftools')
    conf.load('libjson', tooldir='waftools')
    conf.load('trng4', tooldir='waftools')

    conf.check_cxx(cxxflags=['-fopenmp'], ldflags=['-fopenmp'],
                   libpath=['{0}/lib/'.format(conf.options.openmp_root)],
                   uselib_store='OpenMP')

    if conf.options.enable_mpi:
        conf.load('mpi', tooldir='waftools')


def build(bld):
    directories = ['3rd-party', 'include', 'tools', 'test']

    bld.recurse(directories)


def build_docs(bld):
    if bld.env.ENABLE_DOCS:
        bld(features='sphinx', sources='docs')
    else:
        bld.fatal('Please configure with --enable-docs')

from waflib import Build
class docs(Build.BuildContext):
    fun = 'build_docs'
    cmd = 'docs'
