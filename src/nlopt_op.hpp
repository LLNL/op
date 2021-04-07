#pragma once
#include "op.hpp"
#include <nlopt.hpp>

namespace op {

double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective)
{
  auto o = static_cast<op::Functional*>(objective);
  grad   = o->EvalGradient(x);

  return o->Eval(x);
};

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

class NLopt : public op::Optimizer {
public:
  struct Options {
    std::unordered_map<std::string, int>         Int;
    std::unordered_map<std::string, double>      Double;
    std::unordered_map<std::string, std::string> String;
  };

  explicit NLopt(op::Vector<std::vector<double>>& variables, Options& o);

  void setObjective(op::Functional& o) override;
  void addConstraint(op::Functional& o) override;

  // current constraint tolerance
  double constraint_tol;

protected:
  op::Vector<std::vector<double>>& variables_;

  std::unique_ptr<nlopt::opt> nlopt_;
  Options&                    options_;
};

// end NLopt implementation

}  // namespace op
