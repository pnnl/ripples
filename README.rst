Ripples
*******

.. image:: https://zenodo.org/badge/191654750.svg
   :target: https://zenodo.org/badge/latestdoi/191654750

Ripples is a software framework to study the Influence Maximization problem.
The problem of Influence Maximization was introduced in 2001 by Domingos and
Richardson [DR01]_ and later formulated as an optimization problem under the
framework of submodular functions by Kempe et al. [KTR03]_.

Given a graph :math:`G`, a network diffusion model :math:`M`, and positive
integer :math:`k`, the influence maximization problem is the problem of
selecting a set of seeds :math:`S` of cardinality :math:`k` such that
:math:`\mathop{\mathbb{E}}[I(S)]` is maximized, where :math:`I(S)` is the
influence function.

Our goal with Ripples is to provide tools implementing fast and scalable
state-of-the-art approximation algorithms to solve the influence maximization
problem.

.. [DR01] Pedro M. Domingos and Matthew Richardson. 2001. Mining the network
          value of customers. In Proceedings of the seventh ACM SIGKDD
          international conference on Knowledge discovery and data mining, San
          Francisco, CA, USA, August 26-29, 2001. ACM, 57–66.

.. [KTR03] Kempe, D., Kleinberg, J., & Tardos, É. (2003, August). Maximizing the
           spread of influence through a social network. In Proceedings of the
           ninth ACM SIGKDD international conference on Knowledge discovery and
           data mining (pp. 137-146). ACM.


Publications
============

.. [Cluster19] Marco Minutoli, Mahantesh Halappanavar, Ananth Kalyanaraman, Arun
               Sathanur, Ryan Mcclure, Jason McDermott. 2019. Fast and Scalable
               Implementations of Influence Maximization Algorithms. In
               Proceedings of the IEEE Cluster 2019.
.. [ICS2020] Minutoli, Marco, Maurizio Drocco, Mahantesh Halappanavar, Antonino
               Tumeo, and Ananth Kalyanaraman. "cuRipples: influence
               maximization on multi-CPU systems." In Proceedings of the 34th
               ACM International Conference on Supercomputing.

Quickstart with Conan
=====================

First of all we need to set up the Python environment needed.

.. code-block:: shell

   $ pip install --user pipenv
   $ pipenv --three
   $ pipenv install
   $ pipenv shell

Then we need to install dependencies:

.. code-block:: shell

   $ conan create conan/waf-generator user/stable
   $ conan create conan/trng user/stable
   $ conan create conan/metall user/stable
   $ conan create conan/memkind user/stable
   $ conan install --install-folder build .

To enable Memkind or Metall please replace the conan install command with one of:

.. code-block:: shell

   $ conan install --install-folder build . -o memkind=True
   $ conan install --install-folder build . -o metall=True


Now we are ready to configure and build ripples:

.. code-block:: shell

   $ ./waf configure --enable-mpi build_release
   # or without MPI support
   $ ./waf configure build_release

To enable Memkind or Metal configure and build ripples with:

.. code-block:: shell

   $ ./waf configure --enable-mpi --enable-metall build_release
   # or without MPI support
   $ ./waf configure --enable-metall build_release

For Memkind just replace :code:`--enable-metall` with :code:`--enable-memkind`.

In the case you are a Mac OS user, you will need to install the LLVM toolchain
through brew that comes with OpenMP support.  Compiling Ripples than is as
simple as:

.. code-block:: shell

   $ ./waf configure --openmp-root=/usr/local/opt/llvm --enable-mpi build_release
   # or without MPI support
   $ ./waf configure --openmp-root=/usr/local/opt/llvm build_release


Build Instructions
==================

This project uses `WAF <https://waf.io>`_ as its build system.  Building Ripples
is a two-step process: configure the project and build the tools.  Before
attempting to build, be sure to have the following dependencies installed:

- A compiler with C++14 support and OpenMP support.
- `Spdlog <https://github.com/gabime/spdlog>`_
- `JSON <https://github.com/nlohmann/json>`_
- `TRNG4 <https://github.com/rabauke/trng4>`_
- An MPI library (optional)

The configure step can be invoked with:

.. code-block:: shell

   $ ./waf configure

or optionally to enable the MPI implementations:

.. code-block:: shell

   $ ./waf configure --enable-mpi

The build system offers options that can be used to help the configuration step
locate dependencies (e.g., they are installed in unconventional paths).  A
complete list of the options can be obtained with:

.. code-block:: shell

   $ ./waf configure --help


After the configuration step succeeds, the build step can be executed by
running:

.. code-block:: shell

   $ ./waf build_release

For more detailed instruction, please read :ref:`build:Step By Step Build
Instructions`.

The tools compiled can be found under ``build/release/tools/``.  A complete set of
command line options can be obtained through:

.. code-block:: shell

   $ ./build/release/tools/<tool_name> --help


Ripples Team
============

- `Marco Mintutoli <marco.minutoli@pnnl.gov>`_
- `Mahantesh Halappanavar <mahantesh.halappanavar@pnnl.gov>`_
- `Ananth Kalyanaraman <ananth@wsu.edu>`_
- `Maurizio Drocco <maurizio.drocco@ibm.com>`_

Disclamer Notice
================

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or any
information, apparatus, product, software, or process disclosed, or represents
that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

.. raw:: html

   <div align=center>
   <pre style="align-text:center">
   PACIFIC NORTHWEST NATIONAL LABORATORY
   operated by
   BATTELLE
   for the
   UNITED STATES DEPARTMENT OF ENERGY
   under Contract DE-AC05-76RL01830
   </pre>
   </div>
