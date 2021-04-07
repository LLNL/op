#pragma once
#include "op.hpp"
#include <nlopt.hpp>

namespace op {

  /**
   * @brief Takes in a op::Functional and computes the objective function and it's gradient as a nlopt function
   *
   * Has the same signature as nlopt::function so we can convert any op::Functional into a nlopt::function
   * @param[in] x the optimization variables
   * @param[in] grad the result of the gradient of the function w.r.t. x
   * @param[in[ objective A clever way of getting any objective data into a nlopt::function
   */ 
double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective)
{
  auto o = static_cast<op::Functional*>(objective);
  grad   = o->EvalGradient(x);

  return o->Eval(x);
};

  /**
   * @brief wraps any nltop::function into an objective call and a gradient call
   *
   * @param[in] func a nlopt::function
   */
auto wrapNLoptFunc(std::function<double(unsigned, const double*, double*, void*)> func)
{
  auto obj_eval = [&](const std::vector<double>& x) { return func(0, x.data(), nullptr, nullptr); };

  auto obj_grad = [&](const std::vector<double>& x) {
    std::vector<double> grad(x.size());
    func(0, x.data(), grad.data(), nullptr);
    return grad;
  };
  return std::make_tuple<op::Functional::EvalObjectiveFn, op::Functional::EvalObjectiveGradFn>(obj_eval, obj_grad);
}

  /// A op::optimizer implementation for NLopt
class NLopt : public op::Optimizer {
public:

  /// Options specific for nlopt. They are made to look like ipopt's interface
  struct Options {
    std::unordered_map<std::string, int>         Int;
    std::unordered_map<std::string, double>      Double;
    std::unordered_map<std::string, std::string> String;
  };

  /// Constructor for our optimizer
  explicit NLopt(op::Vector<std::vector<double>>& variables, Options& o);

  void setObjective(op::Functional& o) override;
  void addConstraint(op::Functional& o) override;

protected:
  op::Vector<std::vector<double>>& variables_;

  std::unique_ptr<nlopt::opt> nlopt_;
  Options&                    options_;
};

// end NLopt implementation

}  // namespace op
