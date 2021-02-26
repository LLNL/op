#include "op.hpp"
#include <iostream>

class TestOptimizer : public op::Optimizer
{
public:
  TestOptimizer() {
    std::cout << "TestOptimizer Constructor" << std::endl;
  }

  void setObjective(op::Objective &) override {
    std::cout << "Set Objective" << std::endl;
  }

  double Solution() override {
    return 0;
  }
  
};

extern "C" std::unique_ptr<op::Optimizer> load_optimizer(void *)
{
  return std::make_unique<TestOptimizer>();
}
