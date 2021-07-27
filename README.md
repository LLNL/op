# The Optimization Plugin Package (aka "op")

`op` is a lightweight general optimization solver interface. The primary purpose of `op` is to simplify the process of integrating different optimization solvers (serial or parallel) with scalable parallel physics engines. By design, it has several features that help make this a reality.

  * The core abstraction interface was developed to encompass a large class of optimization problems in an optimizer agnostic way. This allows us to describe the optimization problem once and then use a variety of supported `op` optimizers with none to little code-changes.
  *  The `op` core abstract interface is made up of lightweight wrappers that make it easy to integrate with existing simulation codes. This makes integration less intrusive and helps maintain the "integrity" of the physics code. An added benefit is that we can "re-run" an optimization problem on different physics engines by making only a few changes in `op`.
  * The `op` interface includes an assortment of utility methods to perform complicated Halo-exchange patterns based on very little information as well as converting optimization primitives from one framework (e.g. `nlopt` to `ipopt`). In addition, the `op::utility` and `op::mpi` interface is available for users to customize their own MPI patterns as necessary. 
  * Lastly a black-box optimizer plugin interface is provided in `op` to allow for use of proprietary optimization engines without explicit reference in the source code.

The serial `nlopt` optimization library is used to demonstrate how `op` bridges the gap between serial optimizers and parallel simulation codes. `op::NLopt` will run the optimization problem in serial but the physics, objectives, constraints, and respective gradients are computed in parallel. From a user perspective, optimization problems can be described abstractly and `op` guaranatees portability for different optimization engines and optimization problem configurations.

## Install
The following will install to `../install`.
```
mkdir build
cd build
cmake -DBLT_SOURCE_DIR=<blt-dir> -C ../host_configs/quartz.cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install ..

```
## Using OP in another project
The following `cmake` line can be added to `CMakeLists.txt` to find the required package. `OP_DIR` must be defined either as a cmake commandline option, in `CMakeCache.txt`, or in a toolchain file.

```
find_package(op CONFIG REQUIRED
                PATHS ${OP_DIR}/lib/cmake)
```

### Spack package
There is a `spack` package for use with `uberenv` available in `<op-root>/spack/packages/op`. It can be built with `gcc` with vanilla spack, but `clang` support requires some toolchain fixes available in `serac`.

## Documentation
More user documentation is available in sphinx and API documentation in doxygen. To build the documentation type in the following in the build directory:
`make docs`

Spinx documentation is available in `<build-dir>/src/docs/sphinx` or `<install-dir>/docs/sphinx`. Sphinx documentation covers core `op` abstraction concepts, a suggested migration guide in going from an existing optimization solver to `op`'s more general interface, and also documentation for the `TwoCnsts` Rosenbrock problem. Some documentation for the simplified MPI methods is also provided.

Doxygen API documentation is available in `<docs-dir>/sphinx/op_docs/html/doxygen`.

## Included `op` Examples
Currently there are a few examples
- bin/demo - `demo` is a program to test plugin loading with a dummy optimizer call `test_optimizer`.
- tests/TwoCnsts - A set of tests that demonstrate different ways of using `nlopt` on a common two constraint [rosenbrock problem](https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization).
- VariableMap - A set of tests demonstrating a more complicated halo-exchange pattern and how to both use built-in `op` patterns or build custom communication patterns using `op::mpi` and `op::utility` convenience methods.

### `tests/TwoCnsts`
This is a GTest which constains several tests:
- nlopt_serial - A native two parameter nlopt implementation
- nlopt_op - The same test implemented using the `op::NLopt` interface
- nlopt_op_plugin - The same implementation as `nlopt_op`, but using the dynamic plugin interface.
- nlopt_op_mpi - A two-rank mpi implementation of the same problem using `op`'s "advanced" registration procedure.
- nlopt_op_mpi_1 - A single-rank implementation using `op`'s "advanced" communication pattern. The purpose of this example it to make sure the "advanced" pattern can be used as part of migration to the parallel simulation setting.
- nlopt_op_bridge - A "black-box" optimizer example using an externally loaded plugin. The external plugin is a custom implementation on `ipopt`.