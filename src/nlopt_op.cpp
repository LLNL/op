// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC
// All rights reserved.

// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

/**
 * @brief nlopt plugin loading implementation
 *
 * @param[in] variables Optimization variable abstraction
 * @param[in] options op::NLopt option struct
 */
extern "C" std::unique_ptr<op::NLopt<op::nlopt_index_type>> load_optimizer(op::Vector<std::vector<double>>& variables,
                                                                           op::NLoptOptions&                options)
{
  return std::make_unique<op::NLopt<op::nlopt_index_type>>(variables, options);
}
