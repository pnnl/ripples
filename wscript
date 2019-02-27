#! /usr/bin/env python
# encoding: utf-8

VERSION = '0.0.1'
APPNAME = 'ripples'


def options(opt):
    opt.load('compiler_cxx')
    opt.load('doxygen')

    cfg_options = opt.get_option_group('Configuration options')

    opt.load('trng4', tooldir='waftools')
    opt.load('json', tooldir='waftools')
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
    conf.load('doxygen')

    conf.env.CXXFLAGS += ['-std=c++17', '-O3', '-march=native', '-pipe']

    conf.load('spdlog', tooldir='waftools')
    conf.load('json', tooldir='waftools')
    conf.load('trng4', tooldir='waftools')

    conf.check_cxx(cxxflags=['-fopenmp'], ldflags=['-fopenmp'],
                   libpath=['{0}/lib/'.format(conf.options.openmp_root)],
                   uselib_store='OpenMP')

    if conf.options.enable_mpi:
        conf.load('mpi', tooldir='waftools')


def build(bld):
    directories = ['3rd-party', 'include', 'tools', 'tests']

    bld.recurse(directories)


def build_doxy(bld):

    bld(features='subst',
        source='docs/doxygen.conf.in',
        target='docs/doxygen.conf',
        VERSION=VERSION,
        INPUT=bld.path.get_src()
    )

    bld(features='doxygen',
        doxyfile='docs/doxygen.conf',
        target='docs/doxygen.conf.doxy')

    bld(features='subst',
        source='docs/conf.py.in',
        target='docs/html-out/conf.py',
        VERSION=VERSION,
        SRC=bld.path.get_src()
    )

    bld(rule='sphinx-build -c ${TGT} ${SRC} ${TGT}',
        source=bld.path.get_src().find_node('docs'),
        target='docs/html-out')
    bld.add_manual_dependency(
        bld.path.find_node('docs'),
        bld.path.find_node('docs/conf.py.in'))


from waflib import Build
class doxygen(Build.BuildContext):
    fun = 'build_doxy'
    cmd = 'docs'
