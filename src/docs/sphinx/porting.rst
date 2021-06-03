.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC and
.. ## other OP Project Developers. See the top-level COPYRIGHT file for details.
.. ##

==============================================
Porting existing optimization problems to ``op``
==============================================

While adding optimization support to a physics code can already be difficult; transitioning from one optimization interface to another can be equally tedious and painful. ``op`` attempts to greatly mitigate this problem. In the following example we will transition an ``ipopt`` described problem to to the ``op`` interface.

.. note::
   A standard rosenbrock two constraint problem is implemented using the ``nlopt``, ``op``, and ``op`` plugin interface (``NLopt``, ``ipop``) in several different ways including using MPI.


Overview of core ``op`` abstractions
----------------------------------

While `Core optimization abstractions <sphinx/core_abstractions.html>`_ covers details on the information specific abstractions contain, the following is a list of information/methods that is typically required by an ``ipopt`` optimizer. Since ``ipopt`` is a serial optimizer, we can assume that the number of MPI ranks (``nranks = 1``). However the following is provided for generality.

* Number of optimization rank-local variables on each MPI rank (or an optimization variable array)
* The upper and lower bounds on the rank-local optimization variable array on each MPI rank
* The rank-local initial optimization variable state on each MPI rank
* A method to calculate the objective for the problem (this is referred to as the "global objective" in reference to parallel MPI problems)
* A method for calculating the gradient of the objective w.r.t to rank-local optimization variables on each MPI rank
* Methods for constraints and constraint gradients similar to the methods above for objectives
* Bounds for each constraint
* Optimizer options (number of iterations, tolerances, and etc..)
* [Optional] A method for reducing contributions of gradients from different MPI ranks to a specific optimization variable (These are the "owned" variables on each MPI rank).


Suggested steps in porting from the ``ipopt`` interface to ``op``
-------------------------------------------------------------

The following are suggested steps to methodically transition from a specific optimiation interface to the ``op`` interface.

1. Since ``ipopt`` is a serial interface, we will first make sure the problem runs in serial in ``op``. Use the methods/information from the preceeding section directly in ``op`` primitives. Since ``op`` is a lightweight wrapper which can use C++ lambdas to capture references to existing variables and methods this should be relatively straight forward. One would simply call the relevant `ipopt`-related methods within a function (or lambda) and use that to constrcut the ``op`` abstractions. This would apply for lower and upper bounds, and the calculation of objectives, constraints and their gradients.

2. After the `op`-wrapped transition is verified, one can either copy-paste or reimplement the methods in Step 1, but without references to the previous interface's methods (in this case ``ipopt`` methods). This can be done function by function until there is no longer explicit reference to ``ipopt``. We now have implemented the serial problem "agnostically" in ``op``.

3. Now that the problem works in serial, we can transition to running the problem in parallel. The first step is to simply run the serial problem but use `op`'s parallel ``op::Optimizer`` constructors. We can pass in the relevant ``MPI_comm`` to the problem and verify everything is still working properly. Next we can change the serial objective implementation to one which includes a global reductions and test. We can do this for all of the methods above. This tests that our "parallel" implementation at least still works in serial.

4. Lastly we need to determine if our problem falls under the `Simple` or `Advanced` mode problems. For `Simple` mode problems we can try running with 2+ more MPI tasks and verify that the optimization problem produces the same results. For `Advanced` mode problems, we can use the ``op`` utility methods for `AdvancedRegistration` and so forth to provide the necessary information and verify that the problem produces the expected results.

With these 4 steps, we should have a working port from a serial ``ipopt`` interface to a ``op`` interface that leverages parallel computation.
