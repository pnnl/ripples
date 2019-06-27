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
