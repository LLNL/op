#include "op.hpp"

int main(int argc, char*argv[])
{
  if (argc > 1) {
    auto SetupCallback = [](){};
    auto UpdateCallback = [](){};
    auto optimizer = op::PluginOptimizer(argv[1], SetupCallback, UpdateCallback);
    optimizer->Go();
    optimizer->UpdatedVariableCallback();
  }
  return 0;
}
