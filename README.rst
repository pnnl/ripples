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

   $ python -m venv --prompt ripples-dev .venv
   $ source .venv/bin/activate
   $ pip install conan

Then, we set up the conan profile:

.. code-block:: shell

   $ conan profile detect

You can check that the conan has detected the correct compiler by:

.. code-block:: shell

   $ conan profile show

In many cases this will show the correct configuration. Notable exceptions are
systems where you want to use a provided compiler wrapper (e.g., many HPE
machines ship compiler wrappers) or you want to use hipcc to compile the
framework. In that case you want to edit your conan profile file with:

.. code-block:: shell

   $ vim $(conan profile path default)

Here as a reference you can find how to change the profile to use hipcc:

.. code-block:: conf

    [settings]
    arch=x86_64
    build_type=Release
    compiler=clang
    compiler.cppstd=gnu14
    compiler.libcxx=libstdc++11
    compiler.version=15
    os=Linux
    [buildenv]
    *:CC=hipcc
    *:CXX=hipcc

The next step is to install dependencies:

.. code-block:: shell

    $ conan create conan/trng
    $ conan create conan/rocThrust # if compiling with AMD GPU support.
    $ conan create conan/metall    # if compiling with Metall support.
    $ conan install . --build missing
    $ conan install . --build missing -o gpu=amd # for AMD GPU support.

We can now compile ripples:

.. code-block:: shell

    $ conan build .               # CPU only version
    $ conan build . -o gpu=amd    # with AMD GPU support.

To enable Memkind or Metall please replace the conan install command with one of:

Allocate RRRSets Using Metall
=============================

Ripples + Metall has another mode that allocates intermediate data (called RRRSets) using Metall.

To enable the mode, define ENABLE_METALL_RRRSETS macro (e.g., insert ``#define ENABLE_METALL_RRRSETS`` at the beginning of tools/imm.cc).

The storage directory can be specified with ``--rr-store-dir=<PATH>`` argument when executing imm.

Ripples Team
============

- `Marco Mintutoli <marco.minutoli@pnnl.gov>`_
- `Mahantesh Halappanavar <mahantesh.halappanavar@pnnl.gov>`_
- `Ananth Kalyanaraman <ananth@wsu.edu>`_
- `Maurizio Drocco <maurizio.drocco@ibm.com>`_
- `Reece Neff <reece.neff@pnnl.gov>`_

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
