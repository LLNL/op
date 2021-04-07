.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC and
.. ## other OP Project Developers. See the top-level COPYRIGHT file for details.
.. ##

.. _quickstart-label:

======================
Quickstart Guide
======================

Getting OP (The optimization plugin package)
-------------

OP is hosted on `Gitlab <https://lc.llnl.gov/gitlab/wong125/op/>`_. OP uses git submodules, so the project must be cloned recursively. Use either of the following commands to pull Serac's repository:

.. code-block:: bash

   # Using SSH keys setup with GitHub
   $ git clone --recursive ssh://git@czgitlab.llnl.gov:7999/wong125/op.git

Overview of the OP build process
------------------------------------

The op build process has been broken into three phases with various related options:

1. (Optional) Build the developer tools
2. Build the third party libraries
3. Build the serac source code

The developer tools are only required if you wish to contribute to the Serac source code. The first two steps involve building all of the 
third party libraries that are required by Serac. Two options exist for this process: using the `Spack HPC package manager <https://spack.io/>`_
via the `uberenv wrapper script <https://github.com/LLNL/uberenv>`_ or building the required dependencies on your own. We recommend the first
option as building HPC libraries by hand can be a tedious process. Once the third party libraries are built, Serac can be built using the
cmake-based `BLT HPC build system <https://github.com/LLNL/blt>`_.

Building OP
--------------


.. code-block:: bash

   $ mkdir build
   $ cmake -C <toolchain.cmake> -DUSE_NLOPT=ON -D-DNLOPT_DIR=/usr/gapps/transport-opt/transport-tpls/uberenv_libs/gcc-8.3.1/nlopt-2.6.1/ ..

