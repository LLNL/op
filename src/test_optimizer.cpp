// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC
// All rights reserved.

#include "op.hpp"
#include <iostream>

class TestOptimizer : public op::Optimizer {
public:
  TestOptimizer() { std::cout << "TestOptimizer Constructor" << std::endl; }

  void setObjective(op::Functional&) override { std::cout << "Set Objective" << std::endl; }

  double Solution() override { return 0; }
};

extern "C" std::unique_ptr<op::Optimizer> load_optimizer(void*) { return std::make_unique<TestOptimizer>(); }
