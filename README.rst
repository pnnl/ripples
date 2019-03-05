Ripples
*******

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

Quickstart with Docker
======================

.. code-block:: shell

   $ docker pull mminutoli/ripples-dev-env:latest
   $ git clone <url-to-github>
   $ cd ripples
   $ docker-compose -f docker/docker-compose.xml up -d --scale worker=2
   $ docker exec -u mpi -it docker_head_1 /bin/bash
   $ cd $HOME/ripples
   $ ./waf configure --enable-mpi build

To run the parallel IMM tool:

.. code-block:: shell

   $ build/tools/imm  \
        -i test-data/karate.tsv \
        -p \
        -e 0.05 \
        -k 3 \
        -d IC \
        -o output.json

To run the MPI+OpenMP IMM tool on the docker cluster:

.. code-block:: shell

   $  mpiexec --hosts=docker_worker_1,docker_worker_2 -np 2 build/tools/mpi-imm \
        -i test-data/karate.tsv \
        -e 0.05 \
        -k 3 \
        -d IC \
        -o output.json


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

   $ ./waf build

For more detailed instruction, please read :ref:`build:Step By Step Build
Instructions`.

The tools compiled can be found under ``build/tools/``.  A complete set of
command line options can be obtained through:

.. code-block:: shell

   $ ./build/tools/<tool_name> --help
