#!/usr/bin/env python
# encoding: utf-8

"""Tool to generate the Documentation using Sphinx"""

from waflib import Task, Utils, Node
from waflib.TaskGen import feature


class sphinx(Task.Task):
    vars = ['SPHINX_BUILD']
    color = 'BLUE'
    run_str = '${SPHINX_BUILD} ${SRC} ${TGT}'
    always_run = True


@feature('sphinx')
def process_sphinx(self):
    if not getattr(self, 'sources', None):
        self.bld.fatal('Source directory %s not found' % self.sources)

    node = self.sources
    if not isinstance(node, Node.Node):
        node = self.path.find_node(node)
    if not node:
        self.bld.fatal('Source directory %s not found' % self.sources)

    self.create_task('sphinx', node, node.get_bld())


def configure(conf):
    '''Check for doxygen and sphinx'''
    if conf.options.enable_docs:
        conf.load('doxygen')
        conf.load('python')

        conf.find_program('sphinx-build', var='SPHINX_BUILD')

        # check for breathe exhale and sphinx_rtd_theme
        conf.check_python_module('breathe')
        conf.check_python_module('exhale')
        conf.check_python_module('sphinx_rtd_theme')
        conf.env.ENABLE_DOCS = True


def options(opt):
    opt.load('doxygen')
    opt.load('python')

    opt.add_option('--enable-docs', default=False, action='store_true',
                   help='Enable commands to build documentation.')
