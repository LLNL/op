// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

namespace op {
NLopt::NLopt(op::Vector<std::vector<double>>& variables, Options& o)
    : constraint_tol(1.e-8), variables_(variables), options_(o)
{
  std::cout << "NLOpt wrapper constructed" << std::endl;
  nlopt_ = std::make_unique<nlopt::opt>(nlopt::LD_MMA, variables.lowerBounds().size());  // 2 design variables

  // Set variable bounds
  nlopt_->set_lower_bounds(variables.lowerBounds());
  nlopt_->set_upper_bounds(variables.upperBounds());

  // Optimizer settings
  // Process Integer options
  if (o.Int.find("maxeval") != o.Int.end()) nlopt_->set_maxeval(o.Int["maxeval"]);

  // Process Double options
  if (o.Double.find("xtol_rel") != o.Double.end())
    nlopt_->set_xtol_rel(o.Double["xtol_rel"]);  // various tolerance stuff ;)

  // Create default go
  go = [&]() { nlopt_->optimize(variables.data(), final_obj); };
}

void NLopt::setObjective(op::Functional& o) { nlopt_->set_min_objective(NLoptFunctional, &o); }

void NLopt::addConstraint(op::Functional& o)
{
  // Check if constraint_tol key exists in options.Double
  if (options_.Double.find("constraint_tol") == options_.Double.end()) {
    options_.Double["constraint_tol"] = 0.;
  }
  nlopt_->add_inequality_constraint(NLoptFunctional, &o, options_.Double["constraint_tol"]);
};

}  // namespace op
// end NLopt implementation
extern "C" std::unique_ptr<op::NLopt> load_optimizer(op::Vector<std::vector<double>>& variables,
                                                     op::NLopt::Options&              options)
{
  return std::make_unique<op::NLopt>(variables, options);
}
