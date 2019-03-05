Step By Step Build Instructions
*******************************

Install Dependecies
===================

We will assume that the system where you want to run ripples has already
installed a modern C++ compiler with c++14, OpenMP support, and optionally an
MPI library.  If that is not the case, please refer to the relevant
documentation of your operating system or contact your IT department.

.. warning:: The compiler that is part of the Mac OS X Command Line Tools does
             not have support for OpenMP.  We suggest to install the LLVM
             toolchain on your system (e.g., through Homebrew or MacPorts).

Spdlog
------

Ripples requires the Spdlog library to handle the output to console with
detailed timing information of the events during the execution.  In the case
spdlog is not available on your package manager of choice, you can follow these
steps to install spdlog in your system:

.. code-block:: shell

   $ git clone https://github.com/gabime/spdlog
   $ cd spdlog && mkdir build && cd build
   $ cmake -DCMAKE_INSTALL_PREFIX=$SPDLOG_ROOT/root ..
   $ make && make install

where ``$SPDLOG_ROOT`` is the directory where you want the library to be
installed.  Please refer to the `Spdlog <https://github.com/gabime/spdlog>`_
project page to have more detailed information.

JSON
----

Ripples uses JSON files to store the output of its tools.  We use the JSON for
Moder C++ library from Niels Lohman.  In case this library is not available on
your package manager of choice, you can follow these steps to install it on your
system:

.. code-block:: shell

   $ git clone https://github.com/nlohmann/json.git
   $ cd json && mkdir build && cd build
   $ cmake -DCMAKE_INSTALL_PREFIX=$JSON_ROOT/root ..
   $ make && make install

where ``$JSON_ROOT`` is the directory where you want the library to be
installed.  Please refer to the `JSON for modern C++
<https://github.com/nlohmann/json>`_ project page to have more detailed
information.

TRNG4
-----

Random number generation is used extensively in approximation algorithms for the
Influence Maximization problem.  We use the TRNG4 library for our parallel
random number generation.  In the case this library is not available on your
package manager of choice, you can follow these steps to install it on your
system:

.. code-block:: shell

   $ git clone https://github.com/rabauke/trng4.git
   $ cd trng4 && mkdir build
   $ autoreconf -i -f
   $ cd build
   $ ../configure --prefix=$TRNG_ROOT/root
   $ make && make install

where ``$TRNG_ROOT`` is the directory where you want the library to be
installed.  Please refer to the `TRNG4 <https://github.com/rabauke/trng4>`_
project page to have more detailed information.

Build Ripples
=============

Once all the dependencies are installed, Ripples can be configured with:

.. code-block:: shell

   $ ./waf configure --trng4-root=$TRNG_ROOT \
                --spdlog-root=$SPDLOG_ROOT \
                --nlohmann-json-root=$JSON_ROOT \
                --enable-mpi

where ``$TRNG_ROOT``, ``SPDLOG_ROOT``, and ``$JSON_ROOT`` are respectively the
installation directory of TRNG4, Spdlog, and JSON library.

.. note:: Mac OS X user might need to provide the root directory for the OpenMP
          library.  For example after installing LLVM though Homebrew, you need
          to pass the additional ``--openmp-root=/usr/local/opt/llvm`` flag to
          the configure step.

Once the configuration step completes, Ripples can be built by executing:

.. code-block:: shell

   $ ./waf build
