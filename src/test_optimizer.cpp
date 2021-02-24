#include "op.hpp"
#include <iostream>

class TestOptimizer : public op::Optimizer
{
public:
  TestOptimizer(op::CallbackFn setup, op::CallbackFn update) : op::Optimizer(setup, update) {
    std::cout << "TestOptimizer Constructor" << std::endl;
  }

};

extern "C" std::unique_ptr<op::Optimizer> load_optimizer(op::CallbackFn setup, op::CallbackFn update)
{
  return std::make_unique<TestOptimizer>(setup, update);
}
