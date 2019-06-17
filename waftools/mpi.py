#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Battelle Memorial Institute.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tool to detect MPI Libraries.

This tool helps with finding MPI libraries.
"""


from waflib.Configure import conf


def options(opt):
    """Command-line options.
    """
    opt_group = opt.add_option_group('Configuration options')
    for i in 'mpidir mpibin mpilibs'.split():
        opt_group.add_option('--'+i, type=str, default='', dest=i)


def configure(self):
    """
    """
    if self.check_builtin_support():
        return
    self.check_using_mpi_compiler_wrapper()


mpi_cc_sample_snippet = """#include <mpi.h>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  MPI_Finalize();
  return 0;
}
"""


@conf
def check_builtin_support(self):
    result = False
    if self.env.CC:
        result = self.check_c(
            fragment=mpi_cc_sample_snippet,
            execute=False,
            mandatory=False,
            msg = "Checking for MPI compiler builtin support")
    elif self.env.CXX:
        result = self.check_cxx(
            fragment=mpi_cc_sample_snippet,
            execute=False,
            mandatory=False,
            msg = "Checking for MPI compiler builtin support")
    else:
        self.fatal("One between a C and a C++ compiler must be present")
    self.env.HAVE_MPI = result
    return result


@conf
def check_using_mpi_compiler_wrapper(self):
    self.find_program('mpicc', var='MPICC', mandatory=False)
    msg = 'Checking for MPI using mpicc with {0}'

    args_list = { 'OpenMPI' : ['-showme:compile', '-showme:link'],
                  'MPICH' : ['-compile-info', '-link-info'] }

    found = False
    for flags in args_list.values():
        for args in flags:
            check = self.check_cfg(path=self.env.MPICC,
                                   args=args,
                                   msg=msg.format(args),
                                   package='', uselib_store='MPI',
                                   mandatory=False)
            found = found and check
        if found:
            break
