.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC and
.. ## other OP Project Developers. See the top-level COPYRIGHT file for details.
.. ##


=====================================
The two-constraint Rosenbrock Example
=====================================

``op`` provides several implementations of the two-constraint `rosenbrock problem  <https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization>`_ to illustrate the changes needed to `port between different optimizers <sphinx/porting.html>`_ as well as highlighting ways ``op`` helps reduce the complexity in migrating from serial optimization problems to parallel optimization problems.

The following tests can be found in ``tests/TwoCnsts.cpp``:
- nlopt_serial - A native two parameter nlopt implementation
- nlopt_op - The same test implemented using the `op::NLopt` interface
- nlopt_op_plugin - The same implementation as `nlopt_op`, but using the dynamic plugin interface.
- nlopt_op_mpi - A two-rank mpi implementation of the same problem using `op`'s "advanced" registration procedure.
- nlopt_op_mpi_1 - A single-rank implementation using `op`'s "advanced" communication pattern. The purpose of this example it to make sure the "advanced" pattern can be used as part of migration to the parallel simulation setting.
- nlopt_op_bridge - A "black-box" optimizer example using an externally loaded plugin. The external plugin is a custom implementation on `ipopt`.

What follows is a series of commentaries on each of the test to better explain differences and nuances in the implementation. All the examples should yield the same solution and have the same initial optimization variable values as well as the same bounds. In this example/test, we want to find a specific minima at (0,0) so we've bounded the the optimization domain and biased the initial optimization varialbe values to try to get there.

nlopt_serial
------------

The ``nlopt_serial`` test is an implementation of the rosenbrock problem using the open-source library `nlopt <https://nlopt.readthedocs.io/en/latest/>`_.  Specifically the "C" interface to ``nlopt`` was used in this example. As such, many of the implementation functions to evaluate the objectives, constraints, and their gradients live outside of the GTest fixture itself. ``nlopt``-specific bounds and constraint commands are used.

nlopt_op
--------

In this variation of the same test, instead of using ``nlopt``-specific methods, we've transitioned to using ``op``'s more agnostic API. This particular implementation is an example of how one can easily port a ``nlopt`` optimization problem to ``op``-abstracted optimization problem description. For example, ``op::wrapNLoptFunc(...)`` can directly wrap the ``nlopt``-specific functions, and even supplement the nlopt functions with extra debugging information as shown when wrapping the objective.

nlopt_op_plugin
---------------
This implementation is more or less the same as the preceeding one except that the `op::NLopt` has been loaded as a "black-box" plugin instead of being linked directly. This serves as a smoke test for the `nlopt_op_bridge` problem that comes later.


nlopt_op_mpi
------------

This variation re-interprets the rosenbrock problem as a simple 2-rank MPI-problem where rank 0 is reponsible for the first optimization variable, `x`, and rank 1 is responsibly for the second variable, `y`. This is an interesing example as the second-term in the objective relies on both `x` and `y`. This illustrates the "Advanced" data-flow described in `here <sphinx/core_abstractions.html>`_. The objectives and constraints are split into the first and second terms corresponding to the ranks and whether those terms are dependent on `x` or `y`. ``generateReducedLocalGradientFunction()`` is used to automatically handle MPI-communication patterns to collect gradient quantities across relevant MPI-ranks and determine the "owned" portions of the gradient to pass back to the underlying serial ``nlopt`` optimizer. Aside from a few extra calls to ``op::AdvancedRegistration(...)`` and in wrapping rank-local functional evaluations and their gradients to account for MPI-communication, the optimization-related code is more or less the same.

nlopt_op_mpi_1
--------------

This variation provides a MPI communicator to ``op::NLopt`` which tests the "Simple" pipeline but does so with only one rank. This is a useful example since using only one rank in parallel should produce the same results as running the problem in serial! This "Simple" pipeline when using only one rank serves as a "smoke-test" in verifying proper implementation of parallel reductions when porting serial optimization problems to a parallel context.

nlopt_op_bridge
---------------
This test is similar to ``nlopt_op_plugin``, except a custom serial-parallel optimizer based on ``ipopt`` is dynamically loaded as a plugin and used as the underlying optimizer. A special ``Settings`` struct is used to pass data to the custom optimizer; however there is no direct including of headers or linking to the custom optimizer libraries. Additionally, with careful inspection, one will realize that the original ``nlopt``-specific optimizations are wrapped in ``op`` and used by ``ipopt`` in this example. This illusrates the flexibility how using a light-weight abstraction layer like ``op`` can increase code re-use and reduce porting and debugging time when moving from one optimizer to another. In this case, the underlying implementations of the optimiation functions remain the same!
