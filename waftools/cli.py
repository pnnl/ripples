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

"""Tool to detect CLI11."""


def options(opt):
    opt_group = opt.add_option_group('Configuration options')
    opt_group.add_option(
        '--cli-root', action='store', default='/usr',
        help='root directory of the installation of CLI11')

def configure(conf):
    if conf.env.INCLUDES_cli11:
        conf.start_msg('Checking for library CLI11')
        conf.end_msg('yes (by conan)')
        return
    conf.check_cxx(uselib_store='cli11',
                   header_name='CLI/CLI.hpp',
                   includes=['{0}/include/'.format(conf.options.cli_root)],
                   msg='Checking for library cli11')
