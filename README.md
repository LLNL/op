# The Optimization Package (aka "op")

The goal is to provide an abstract optimization interface to easily hook into a variety of available optimizers. A plugin interface is included to enable dynamically loaded optimizer implementations.

## Install
The following will install to `../install`.
```
mkdir build
cd build
cmake -DBLT_SOURCE_DIR=<blt-dir> -C ../host_configs/quartz.cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install ..

```

## Test Binaries
Currently there are the finally executables:
- demo
- TwoCnsts

### demo
`demo` is a program to test plugin loading with a dummy optimizer call `test_optimizer`.

### TwoCnsts
This is a GTest which constains several tests:
- nlopt_serial - A native two parameter nlopt implementation
- nlopt_op - The same test implemented using the `op::NLopt` interface
- nlopt_op_plugin - The same implementation as `nlopt_op`, but using the dynamic plugin interface.
