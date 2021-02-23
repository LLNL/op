#include "op.hpp"
#include <iostream>

class TestOptimizer : public op::Optimizer
{
public:
  TestOptimizer(op::CallbackFn setup) : op::Optimizer(setup) {
    std::cout << "TestOptimizer Constructor" << std::endl;
  }

  void UpdatedVariableCallback() override {
    std::cout << "UpdatedVariableCallback" << std::endl;
  }

};

extern "C" std::unique_ptr<op::Optimizer> load_optimizer(op::CallbackFn setup)
{
  return std::make_unique<TestOptimizer>(setup);
}
