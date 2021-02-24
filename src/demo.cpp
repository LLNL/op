#include "op.hpp"

int main(int argc, char*argv[])
{
  if (argc > 1) {
    auto optimizer = op::PluginOptimizer<op::Optimizer>(argv[1]);
    optimizer->Go();
    optimizer->UpdatedVariableCallback();
  }
  return 0;
}
