#pragma once
#include "op.hpp"
#include <nlopt.hpp>

/// Op namespace
namespace op {

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

/// A op::optimizer implementation for NLopt
class NLopt : public op::Optimizer {
public:

  /**
   *  Define a simple state messaging scheme
   *
   * n = constraint.size()
   *
   * -4 => solution found
   * -3 => update variables
   * -2 => obj_grad
   * -1 => obj_eval
   * [0,n-1] => constraint_eval
   * [n,2n-1] => constraint_eval
   *
   */

  enum State : int {
    OTHER,
    SOLUTION_FOUND = -4,
    UPDATE_VARIABLES = -3,
    OBJ_GRAD = -2,
    OBJ_EVAL = -1
  };
  
  /// Container to pass objective and optimizer
  struct FunctionalInfo {
    op::Functional & obj;
    op::NLopt & nlopt;
    int state;
  };

  /// Options specific for nlopt. They are made to look like ipopt's interface
  struct Options {
    std::unordered_map<std::string, int>         Int;
    std::unordered_map<std::string, double>      Double;
    std::unordered_map<std::string, std::string> String;
  };
  
  /// Constructor for our optimizer
  explicit NLopt(op::Vector<std::vector<double>>& variables, Options& o, MPI_Comm comm = MPI_COMM_WORLD);

  void setObjective(op::Functional& o) override;
  void addConstraint(op::Functional& o) override;

  /**
   * @brief Method to see if variables changed, if they have set new x
   *
   * @param[in] x 
   *
   */
  bool variables_changed(const std::vector<double> & x) {
    for (std::size_t i = 0 ; i < x.size(); i++) {
      if (previous_variables_[i] != x[i]) {
	std::copy(x.begin(), x.end(), previous_variables_.begin());
	return false;
      }      
    }
    std::copy(x.begin(), x.end(), previous_variables_.begin());
    return true;
  }

  
protected:
  MPI_Comm comm_;
  std::vector<double> global_variables_;
  op::Vector<std::vector<double>>& variables_;

  std::unique_ptr<nlopt::opt> nlopt_;
  Options&                    options_;

  std::vector<double> previous_variables_;

  std::vector<FunctionalInfo> obj_info_;
  std::vector<FunctionalInfo> constraints_info_;

  std::vector<int> variables_per_rank_;
  std::vector<int> offsets_;

  friend double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective_and_optimizer);

  
};
// end NLopt implementation
  
  
/**
 * @brief Takes in a op::Functional and computes the objective function and it's gradient as a nlopt function
 *
 * Has the same signature as nlopt::function so we can convert any op::Functional into a nlopt::function
 * @param[in] x the optimization variables
 * @param[in] grad the result of the gradient of the function w.r.t. x
 * @param[in] objective Get FunctionalInfo into this call
 */
double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective_and_optimizer)
{
  auto info = static_cast<op::NLopt::FunctionalInfo *>(objective_and_optimizer);
  auto & optimizer = info->nlopt;
  auto & objective = info->obj;
  auto rank = op::mpi::getRank(optimizer.comm_);

  if (rank == 0) {
    if (optimizer.variables_changed(x)) {
      // first thing to do is broadcast the state
      std::vector<int> state{op::NLopt::State::UPDATE_VARIABLES};
      op::mpi::Broadcast(state, 0, optimizer.comm_);
      
      // have the root rank scatter variables back to "owning nodes"
      std::vector<double> x_temp(x.begin(), x.end());
      op::mpi::Scatterv(x_temp, optimizer.variables_per_rank_, optimizer.offsets_, optimizer.variables_.data(), 0, optimizer.comm_);
      optimizer.UpdatedVariableCallback();
    }

    // Next check if gradient needs to be called
    if (grad.size() > 0 ) {
      // check to see if it's an objective or constraint
      std::vector<int> state {info->state < 0 ? op::NLopt::State::OBJ_GRAD : info->state + static_cast<int>(optimizer.constraints_info_.size())};
      op::mpi::Broadcast(state, 0, optimizer.comm_);
    } else {
      // just eval routine
      // check to see if it's an objective or constraint
      std::vector<int> state {info->state < 0 ? op::NLopt::State::OBJ_EVAL : info->state};     
      op::mpi::Broadcast(state, 0, optimizer.comm_);
    }
  } 

  // At this point, the non-rank roots should be calling this method and we should get here
  
  if (grad.size() > 0) {    
    grad   = objective.EvalGradient(x);
  }
  return objective.Eval(x);
};

}  // namespace op
