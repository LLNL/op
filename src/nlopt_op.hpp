#pragma once
#include "op.hpp"
#include <nlopt.hpp>
#include "op_waitloop.hpp"

/// Op namespace
namespace op {

// forward declarations
template <typename T>
class NLopt;
template <typename T>
double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective_and_optimizer);

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

template <typename T>
NLopt(op::Vector<std::vector<double>>, NLoptOptions&, MPI_Comm, utility::CommPattern<T>) -> NLopt<T>;

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
double NLoptFunctional(const std::vector<double>& x, std::vector<double>& grad, void* objective_and_optimizer)
{
  auto  info      = static_cast<op::detail::FunctionalInfo<T>*>(objective_and_optimizer);
  auto& optimizer = info->nlopt.get();
  auto& objective = info->obj.get();

  // check if the optimizer is running in "serial" or "parallel"
  if (optimizer.comm_ == MPI_COMM_NULL) {
    // the optimizer is running in serial
    if (optimizer.variables_changed(x)) {
      optimizer.variables_.data() = x;
      optimizer.UpdatedVariableCallback();
    }
  } else {
    // the optimizer is running in parallel
    auto rank = op::mpi::getRank(optimizer.comm_);

    if (rank == optimizer.root_rank_) {
      if (optimizer.variables_changed(x)) {
        // first thing to do is broadcast the state
        std::vector<int> state{op::State::UPDATE_VARIABLES};
        op::mpi::Broadcast(state, 0, optimizer.comm_);

        // have the root rank scatter variables back to "owning nodes"
        std::vector<double> x_temp(x.begin(), x.end());
        std::vector<double> new_data(optimizer.comm_pattern_.has_value()
                                         ? optimizer.comm_pattern_.value().owned_variable_list.get().size()
                                         : optimizer.variables_.data().size());
        op::mpi::Scatterv(x_temp, optimizer.owned_variables_per_rank_, optimizer.owned_offsets_, new_data, 0,
                          optimizer.comm_);

        if (optimizer.comm_pattern_.has_value()) {
          // repropagate back to non-owning ranks
          optimizer.variables_.data() =
              op::ReturnLocalUpdatedVariables(optimizer.comm_pattern_.value().rank_communication.get(),
                                              optimizer.global_reduced_map_to_local_.value(), new_data);
        } else {
          optimizer.variables_.data() = new_data;
        }

        optimizer.UpdatedVariableCallback();
      }

      // Next check if gradient needs to be called
      if (grad.size() > 0) {
        // check to see if it's an objective or constraint
        std::vector<int> state{info->state < 0 ? op::State::OBJ_GRAD
                                               : info->state + static_cast<int>(optimizer.constraints_info_.size())};
        op::mpi::Broadcast(state, 0, optimizer.comm_);
      } else {
        // just eval routine
        // check to see if it's an objective or constraint
        std::vector<int> state(1, info->state < 0 ? op::State::OBJ_EVAL : info->state);
        op::mpi::Broadcast(state, 0, optimizer.comm_);
      }
    }
  }

  // At this point, the non-rank roots should be calling this method and we should get here
  if (grad.size() > 0) {
    auto owned_grad = objective.EvalGradient(optimizer.variables_.data());

    // check if "serial" or parallel. If we're running in serial we're already done.
    if (optimizer.comm_ != MPI_COMM_NULL) {
      grad = op::utility::parallel::concatGlobalVector(optimizer.owned_offsets_.back(),
                                                       optimizer.owned_variables_per_rank_, optimizer.owned_offsets_,
                                                       owned_grad, false, 0, optimizer.comm_);  // gather on rank 0
    } else {
      grad = owned_grad;
    }

    // check if this is a lower_bound constraint and negate gradient
    if (info->state >= 0 && info->lower_bound) {
      for (auto& g : grad) {
        g *= -1.;
      }
    }
  }

  // modify objective evaluation just for constraints
  if (info->state >= 0) {
    if (info->lower_bound) {
      /**
       * for constraints g >= lower_bound, they need to be rewritten as
       * -(g - lower_bound) <= 0
       */
      return -(objective.Eval(optimizer.variables_.data()) - info->constraint_val);
    }
    /**
     * for constraints g <= upper_bound, they need to be rewritten as
     * g - upper_bound < = 0
     */

    return objective.Eval(optimizer.variables_.data()) - info->constraint_val;
  }

  return objective.Eval(optimizer.variables_.data());
};

/* Parallel simulation , serial optimizer pattern implementation
 *
 */

/// Serial-non-rank wait loop
template <typename T>
std::function<void()> serialOptimizerNonRootWaitLoop(
    const std::function<void()>& update, const std::function<void()>& obj_grad, const std::function<void()>& obj_eval,
    const std::function<void(int)>& constraints_states, const std::function<void(int)>& constraints_grad_states,
    const std::function<void()>& solution_state, const std::function<void(int)>& unknown_state,
    std::vector<detail::FunctionalInfo<T>>& constraints_info, double& final_obj, MPI_Comm comm = MPI_COMM_WORLD)
{
  return [=, &final_obj, &constraints_info]() {
    // we won't know the real number of constraints until the problem actually starts to run
    int nconstraints = constraints_info.size();
    // non-root ranks will be in a perpetual wait loop until we find a solution
    while (final_obj == std::numeric_limits<double>::max()) {
      // set up to recieve
      std::vector<int> state(1);
      op::mpi::Broadcast(state, 0, comm);

      auto opt_state = state.front();

      if (opt_state == op::SOLUTION_FOUND) {
        solution_state();
        break;
      } else if (opt_state >= nconstraints && opt_state < nconstraints * 2) {
        constraints_grad_states(opt_state % nconstraints);
      } else if (opt_state >= 0 && opt_state < nconstraints) {
        constraints_states(opt_state);
      } else {
        switch (opt_state) {
          case op::State::UPDATE_VARIABLES:
            update();
            break;
          case op::State::OBJ_GRAD:
            obj_grad();
            break;
          case op::State::OBJ_EVAL:
            obj_eval();
            break;
          default:
            unknown_state(opt_state);
            break;
        }
      }
    }
  };
}

}  // namespace op
