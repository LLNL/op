.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC and
.. ## other OP Project Developers. See the top-level COPYRIGHT file for details.
.. ##


=======
OP (The optimization plugin package)
=======

OP is a general optimization solver interface. It's primary purpose is to faciliate experimenting with different optimization solvers. It allows users to writing new objective functions in a general and abstract framework, and helps convert objective functions hard-coded for other optimization interfaces and using them with a general set of optimizers. It also contains an MPI-abstraction interface library as well as additional MPI-patterns to make it significantly easier to perform halo-exchanges necessary in parallelizing optimization problems.


  *  :ref:`Quickstart/Build Instructions <quickstart-label>`
  *  `Source Documentation <doxygen/html/index.html>`_


======================================================
Copyright and License Information
======================================================

Copyright (c) 2021, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

.. toctree::
   :hidden:
   :maxdepth: 2

   sphinx/quickstart
