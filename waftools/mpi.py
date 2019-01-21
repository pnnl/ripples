#!/usr/bin/env python
# encoding: utf-8

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
  MPI_Init(argc, argv);
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
