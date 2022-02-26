// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC
// All rights reserved.

#include "op.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc > 1) {
    auto optimizer = op::PluginOptimizer<op::Optimizer>(argv[1]);
    if (optimizer) {
      optimizer->Go();
      optimizer->UpdatedVariableCallback();
    } else {
      std::cout << "The optimizer did not load properly:" << argv[1] << std::endl;
    }
  }
  return 0;
}
