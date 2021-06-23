.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC and
.. ## other OP Project Developers. See the top-level COPYRIGHT file for details.
.. ##


====================================
op (The optimization plugin package)
====================================

`op` is a lightweight general optimization solver interface. The primary purpose of `op` is to simplify the process of integrating different optimization solvers (serial or parallel) with scalable parallel physics engines. By design, it has several features that help make this a reality.

  * The core abstraction interface was developed to encompass a large class of optimization problems in an optimizer agnostic way. This allows us to describe the optimization problem once and then use a variety of supported `op` optimizers with none to little code-changes.
  *  The `op` core abstract interface is made up of lightweight wrappers that make it easy to integrate with existing simulation codes. This makes integration less intrusive and helps maintain the "integrity" of the physics code. An added benefit is that we can "re-run" an optimization problem on different physics engines by making only a few changes in `op`.
  * The `op` interface includes an assortment of utility methods to perform complicated Halo-exchange patterns based on very little information as well as converting optimization primitives from one framework (e.g. `nlopt` to `ipopt`). In addition, the `op::utility` and `op::mpi` interface is available for users to customize their own MPI patterns as necessary. 
  * Lastly a black-box optimizer plugin interface is provided in `op` to allow for use of proprietary optimization engines without explicit reference in the source code.

The serial `nlopt` optimization library is used to demonstrate how `op` bridges the gap between serial optimizers and parallel simulation codes. `op::NLopt` will run the optimization problem in serial but the physics, objectives, constraints, and respective gradients are computed in parallel. From a user perspective, optimization problems can be described abstractly and `op` guaranatees portability for different optimization engines and optimization problem configurations.
    
=============
Documentation
=============

  *  :ref:`Quickstart/Build Instructions <quickstart-label>`
  *  `Source Documentation <doxygen/html/index.html>`_
  *  `Core optimization abstractions <sphinx/core_abstractions.html>`_
  *  `Porting existing optimization problems to the op interface <sphinx/porting.html>`_
  
  
======================================================
Copyright and License Information
======================================================

Copyright (c) 2021, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

.. toctree::
   :hidden:
   :maxdepth: 2

   sphinx/quickstart
   sphinx/core_abstractions
   sphinx/porting
