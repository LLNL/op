#pragma once
#include "op.hpp"
#include <nlopt.hpp>
#include "op_waitloop.hpp"

/// Op namespace
namespace op {

/**
 * @brief Takes in a op::Functional and computes the objective function and it's gradient as a nlopt function
 *
 * Has the same signature as nlopt::function so we can convert any op::Functional into a nlopt::function
 * @param[in] x the optimization variables (on rank = 0 this is the actual global optimization variables, on other ranks
 * it is the local-view of variables.data())
 * @param[in] grad the result of the gradient of the function w.r.t. x (on rank 0, this is the global gradient eval, on
 * other ranks it is the owned-local gradient)
 * @param[in] objective Get FunctionalInfo into this call
 */

template <typename T>
double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective_and_optimizer);

// forward declarations for NLopt class
template <typename T>
class NLopt;

// detail namespace
namespace detail {
/// Container to pass objective and optimizer
template <typename T>
struct FunctionalInfo {
  std::reference_wrapper<op::Functional> obj;
  std::reference_wrapper<op::NLopt<T>>   nlopt;  // TODO: probably should just template op::NLopt<T>
  int                                    state;
  double                                 constraint_tol = 0.;
  double                                 constraint_val = 0.;
  bool                                   lower_bound    = false;
};

}  // namespace detail

/// Default nlopt type
using nlopt_index_type = std::vector<std::size_t>;

/**
 * @brief wraps any nltop::function into an objective call and a gradient call
 *
 * @param[in] func a nlopt::function
 */
auto wrapNLoptFunc(std::function<double(unsigned, const double*, double*, void*)> func)
{
  auto obj_eval = [&](const std::vector<double>& x) { return func(x.size(), x.data(), nullptr, nullptr); };

  auto obj_grad = [&](const std::vector<double>& x) {
    std::vector<double> grad(x.size());
    func(x.size(), x.data(), grad.data(), nullptr);
    return grad;
  };
  return std::make_tuple<op::Functional::EvalObjectiveFn, op::Functional::EvalObjectiveGradFn>(obj_eval, obj_grad);
}

/// Options specific for nlopt. They are made to look like ipopt's interface
struct NLoptOptions {
  std::unordered_map<std::string, int>         Int;
  std::unordered_map<std::string, double>      Double;
  std::unordered_map<std::string, std::string> String;
  nlopt::algorithm                             algorithm = nlopt::LD_MMA;
};

/// A op::optimizer implementation for NLopt
template <typename T = nlopt_index_type>
class NLopt : public op::Optimizer {
public:
  /// Constructor for our optimizer
  explicit NLopt(op::Vector<std::vector<double>>& variables, NLoptOptions& o, std::optional<MPI_Comm> comm = {},
                 std::optional<op::utility::CommPattern<T>> comm_pattern = {});

  void setObjective(op::Functional& o) override;
  void addConstraint(op::Functional& o) override;

  /**
   * @brief Method to see if variables changed, if they have set new x
   *
   * @param[in] x
   *
   */
  bool variables_changed(const std::vector<double>& x)
  {
    assert(x.size() == previous_variables_.size());
    for (std::size_t i = 0; i < x.size(); i++) {
      if (previous_variables_[i] != x[i]) {
        std::copy(x.begin(), x.end(), previous_variables_.begin());
        return true;
      }
    }
    return false;
  }

protected:
  MPI_Comm                         comm_;
  std::vector<double>              global_variables_;
  op::Vector<std::vector<double>>& variables_;

  std::unique_ptr<nlopt::opt> nlopt_;
  NLoptOptions&               options_;

  std::vector<double> previous_variables_;

  std::vector<detail::FunctionalInfo<T>> obj_info_;
  std::vector<detail::FunctionalInfo<T>> constraints_info_;

  std::vector<int> owned_variables_per_rank_;
  std::vector<int> owned_offsets_;

  std::optional<utility::CommPattern<T>> comm_pattern_;

  std::optional<std::unordered_map<typename T::value_type, T>> global_reduced_map_to_local_;

  friend double NLoptFunctional<T>(const std::vector<double>& x, std::vector<double>& grad,
                                   void* objective_and_optimizer);

  std::size_t               num_local_owned_variables_;
  int                       root_rank_ = 0;
  std::unique_ptr<WaitLoop> waitloop_;
};
// end NLopt implementation

// template deduction guide
template <typename T>
NLopt(op::Vector<std::vector<double>>, NLoptOptions&, MPI_Comm, utility::CommPattern<T>) -> NLopt<T>;

}  // namespace op
