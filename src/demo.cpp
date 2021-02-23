#include "op.hpp"

int main(int argc, char*argv[])
{
  if (argc > 1) {
    auto SetupCallback = [](){};
    auto optimizer = op::PluginOptimizer(argv[1], SetupCallback);
    optimizer->UpdatedVariableCallback();
  }
  return 0;
}
