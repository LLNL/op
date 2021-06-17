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
