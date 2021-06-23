#pragma once
#include "op.hpp"
#include <nlopt.hpp>
#include "op_waitloop.hpp"

/// Op namespace
namespace op {

// forward declaration of NLoptFunctional
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
  op::Functional obj;  // we don't use a reference here incase we need to wrap obj
  op::NLopt<T>&  nlopt;
  int            state;
  double         constraint_tol = 0.;
  double         constraint_val = 0.;
  bool           lower_bound    = false;
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
  auto obj_eval = [&](const std::vector<double>& x) {
    return func(static_cast<unsigned int>(x.size()), x.data(), nullptr, nullptr);
  };

  auto obj_grad = [&](const std::vector<double>& x) {
    std::vector<double> grad(x.size());
    func(static_cast<unsigned int>(x.size()), x.data(), grad.data(), nullptr);
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
                 std::optional<op::utility::CommPattern<T>> comm_pattern_info = {})
      : comm_(comm.has_value() ? comm.value() : MPI_COMM_NULL),
        global_variables_(0),
        variables_(variables),
        options_(o),
        comm_pattern_(comm_pattern_info),
        global_reduced_map_to_local_({}),
        num_local_owned_variables_(0)
  {
    std::cout << "NLOpt wrapper constructed" << std::endl;

    // if we want to run nlopt in "serial" mode, set rank = 0, otherwise get the rank in the communicator
    auto rank = comm_ != MPI_COMM_NULL ? op::mpi::getRank(comm_) : 0;

    // Since this optimizer runs in serial we need to figure out the global number of decisionvariables
    // in "advanced" mode some of the local_variables may not be unique to each partition
    if (comm_pattern_.has_value()) {
      // figure out what optimization variables we actually own
      auto& comm_pattern           = comm_pattern_.value();
      num_local_owned_variables_   = comm_pattern.owned_variable_list.size();
      global_reduced_map_to_local_ = op::utility::inverseMap(comm_pattern.local_variable_list);
    } else {
      num_local_owned_variables_ = variables.data().size();
    }

    // in "simple" mode the local_variables are unique to each partition, but by the time
    // we get here both the "advanced" and "simple" pattern is the same for calculating
    // the actual optimization problem size
    if (comm_ != MPI_COMM_NULL) {
      auto [global_size, variables_per_rank] =
          op::utility::parallel::gatherVariablesPerRank<int>(num_local_owned_variables_);
      owned_variables_per_rank_ = variables_per_rank;
      owned_offsets_            = op::utility::buildInclusiveOffsets(owned_variables_per_rank_);

      if (rank == root_rank_) {
        global_variables_.resize(static_cast<std::size_t>(global_size));
      }
    } else {
      global_variables_.resize(variables_.data().size());
    }

    // Create nlopt optimizer
    nlopt_ = std::make_unique<nlopt::opt>(options_.algorithm, global_variables_.size());

    // Set variable bounds
    auto lowerBounds = variables.lowerBounds();
    auto upperBounds = variables.upperBounds();

    // Adjust in "advanced" mode
    if (isAdvanced()) {
      auto& reduced_variable_list = comm_pattern_.value().owned_variable_list;
      lowerBounds =
          op::utility::permuteMapAccessStore(lowerBounds, reduced_variable_list, global_reduced_map_to_local_.value());
      upperBounds =
          op::utility::permuteMapAccessStore(upperBounds, reduced_variable_list, global_reduced_map_to_local_.value());
    }

    // save initial set of variables to detect if variables changed
    // set previous_variables to make the check
    if (isAdvanced()) {
      auto& reduced_variable_list            = comm_pattern_.value().owned_variable_list;
      auto  reduced_previous_local_variables = op::utility::permuteMapAccessStore(
          variables.data(), reduced_variable_list, global_reduced_map_to_local_.value());
      previous_variables_ =
          op::utility::parallel::concatGlobalVector(global_variables_.size(), owned_variables_per_rank_, owned_offsets_,
                                                    reduced_previous_local_variables, false);  // gather on rank 0

    } else if (comm_ != MPI_COMM_NULL) {
      // in this case everyone owns all of their variables
      previous_variables_ =
          op::utility::parallel::concatGlobalVector(global_variables_.size(), owned_variables_per_rank_, owned_offsets_,
                                                    variables.data(), false);  // gather on rank 0
    } else {
      // "serial" case
      previous_variables_ = variables.data();
    }

    // initialize our starting global variables
    global_variables_ = previous_variables_;

    if (comm_ != MPI_COMM_NULL) {
      auto global_lower_bounds = op::utility::parallel::concatGlobalVector(
          global_variables_.size(), owned_variables_per_rank_, owned_offsets_, lowerBounds, false);  // gather on rank 0

      auto global_upper_bounds = op::utility::parallel::concatGlobalVector(
          global_variables_.size(), owned_variables_per_rank_, owned_offsets_, upperBounds, false);  // gather on rank 0
      if (rank == root_rank_) {
        nlopt_->set_lower_bounds(global_lower_bounds);
        nlopt_->set_upper_bounds(global_upper_bounds);
      }
    } else {
      // in the serial case we know this rank is root already
      nlopt_->set_lower_bounds(lowerBounds);
      nlopt_->set_upper_bounds(upperBounds);
    }

    // Optimizer settings only need to be set on the root rank as it runs the actual NLopt optimizer
    if (rank == root_rank_) {
      // Process Integer options
      if (o.Int.find("maxeval") != o.Int.end()) nlopt_->set_maxeval(o.Int["maxeval"]);

      // Process Double options
      for (auto [optname, optval] : options_.Double) {
        if (optname == "xtol_rel") {
          nlopt_->set_xtol_rel(o.Double["xtol_rel"]);  // various tolerance stuff ;)
        } else {
          nlopt_->set_param(optname.c_str(), optval);
        }
      }

      // Check if constraint_tol key exists in options.Double
      if (options_.Double.find("constraint_tol") == options_.Double.end()) {
        options_.Double["constraint_tol"] = 0.;
      }
    }

    // Create the go routine to start the optimization

    if (rank == root_rank_) {
      // The root_rank uses nlopt to perform the optimization.
      // Within NLoptFunctional there is a branch condition for the root_rank which broadcasts
      // the current state to all non-root ranks.

      go.onGo([&]() {
        nlopt_->set_min_objective(NLoptFunctional<T>, &obj_info_[0]);
        for (auto& constraint : constraints_info_) {
          nlopt_->add_inequality_constraint(NLoptFunctional<T>, &constraint, constraint.constraint_tol);
        }

        nlopt_->optimize(global_variables_, final_obj);

        // propagate solution objective to all ranks
        if (comm_ != MPI_COMM_NULL) {
          std::vector<int> state{op::State::SOLUTION_FOUND};
          op::mpi::Broadcast(state, 0, comm_);

          std::vector<double> obj(1, final_obj);
          op::mpi::Broadcast(obj, 0, comm_);
        }
      }

      );
    } else {
      // Non-root ranks go into a wait loop where they wait to recieve the current state
      // If evaluation calls are made `NLoptFunctional` is called

      waitloop_ = std::make_unique<WaitLoop>([&]() { return constraints_info_.size(); }, final_obj, comm_);

      // Add actions customized for NLopt to the waitloop-Fluent pattern

      (*waitloop_)
          .onUpdate([&, rank]() {
            // recieve the incoming variables
            std::vector<double> owned_data(num_local_owned_variables_);
            std::vector<double> empty;
            op::mpi::Scatterv(empty, owned_variables_per_rank_, owned_offsets_, owned_data, 0, comm_);
            if (comm_pattern_.has_value()) {
              // repropagate back to non-owning ranks
              std::vector<double> local_data(variables_.data().size());
              auto&               owned_variable_list = comm_pattern_.value().owned_variable_list;
              std::cout << "update: " << rank << " " << owned_data.size() << " " << owned_variable_list.size() << " "
                        << local_data.size() << " ";
              for (std::size_t i = 0; i < owned_data.size(); i++) {
                std::cout << owned_variable_list[i] << ":" << owned_data[i] << " ";
              }
              std::cout << std::endl;

              // TODO: improve fix during refactor
              std::vector<typename T::value_type> index_map;
              for (auto id : comm_pattern_.value().owned_variable_list) {
                index_map.push_back(global_reduced_map_to_local_.value()[id][0]);
              }
	      
              op::utility::accessPermuteStore(owned_data, index_map, local_data);

              variables_.data() = op::ReturnLocalUpdatedVariables(comm_pattern_.value().rank_communication,
                                                                  global_reduced_map_to_local_.value(), local_data);
            } else {
              variables_.data() = owned_data;
            }

            UpdatedVariableCallback();
          })
          .onObjectiveGrad(
              // obj_grad
              [&]() {
                std::vector<double> grad(variables_.data().size());
                // Call NLoptFunctional on non-root-rank
                NLoptFunctional<T>(variables_.data(), grad, &obj_info_[0]);
              })
          .onObjectiveEval(
              // obj_Eval
              [&]() {
                std::vector<double> grad;
                // Call NLoptFunctional on non-root-rank
                NLoptFunctional<T>(variables_.data(), grad, &obj_info_[0]);
              })
          .onConstraintsEval(
              // constraint states
              [&](int state) {
                // just an eval routine
                std::vector<double> grad;
                // Call NLoptFunctional on non-root-rank
                NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[static_cast<std::size_t>(state)]);
              })
          .onConstraintsGrad(
              // constraint grad states
              [&](int state) {
                // this is a constraint gradient call
                std::vector<double> grad(variables_.data().size());
                // Call NLoptFunctional on non-root-rank
                NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[static_cast<std::size_t>(state)]);
              })
          .onSolution(  // Solution state
              [&]() {
                // The solution has been found.. recieve the objective
                std::vector<double> obj(1);
                op::mpi::Broadcast(obj, 0, comm_);
                final_obj = obj[0];
                // exit this loop finally!
              })
          .onUnknown(  // unknown state
              [&](int state) {
                // this is an unknown state!
                std::cout << "Unknown State: " << state << std::endl;
                MPI_Abort(comm_, state);
              });

      // Set the Go function to use the waitloop functor we've just created
      go.onGo([&]() { (*waitloop_)(); });
    }
  }

  void setObjective(op::Functional& o) override
  {
    obj_info_.clear();
    obj_info_.emplace_back(
        op::detail::FunctionalInfo<T>{.obj = o, .nlopt = *this, .state = State::OBJ_EVAL, .constraint_tol = 0.});
  }

  void addConstraint(op::Functional& o) override
  {
    if (o.upper_bound != op::Functional::default_max) {
      constraints_info_.emplace_back(op::detail::FunctionalInfo<T>{.obj   = o,
                                                                   .nlopt = *this,
                                                                   .state = static_cast<int>(constraints_info_.size()),
                                                                   .constraint_tol = options_.Double["constraint_tol"],
                                                                   .constraint_val = o.upper_bound,
                                                                   .lower_bound    = false});
    }
    if (o.lower_bound != op::Functional::default_min) {
      constraints_info_.emplace_back(op::detail::FunctionalInfo<T>{.obj   = o,
                                                                   .nlopt = *this,
                                                                   .state = static_cast<int>(constraints_info_.size()),
                                                                   .constraint_tol = options_.Double["constraint_tol"],
                                                                   .constraint_val = o.lower_bound,
                                                                   .lower_bound    = true});
    }
  }

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

  /**
   * @brief returns whether NLopt is in "advanced" mode or not
   */
  bool isAdvanced() { return comm_pattern_.has_value(); }

  /**
   * @brief generates reduced local gradient using comm_pattern_
   */
  auto generateReducedLocalGradientFunction(
      std::function<std::vector<double>(const std::vector<double>&)> local_grad_func,
      std::function<double(const std::vector<double>&)>              local_reduce_func)
  {
    assert(comm_pattern_.has_value());
    return op::OwnedLocalObjectiveGradientFunction(
        comm_pattern_.value().rank_communication, global_reduced_map_to_local_.value(),
        comm_pattern_.value().owned_variable_list, local_grad_func, local_reduce_func, comm_);
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

  std::vector<int> owned_variables_per_rank_;  // this needs to be `int` to satisify MPI
  std::vector<int> owned_offsets_;             // this needs to be `int` to satisfy MPI

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
  auto& optimizer = info->nlopt;
  auto& objective = info->obj;

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
                                         ? optimizer.comm_pattern_.value().owned_variable_list.size()
                                         : optimizer.variables_.data().size());
        op::mpi::Scatterv(x_temp, optimizer.owned_variables_per_rank_, optimizer.owned_offsets_, new_data, 0,
                          optimizer.comm_);

        if (optimizer.comm_pattern_.has_value()) {
          // repropagate back to non-owning ranks
          std::vector<double> local_data(optimizer.variables_.data().size());
          auto&               owned_variable_list = optimizer.comm_pattern_.value().owned_variable_list;
          std::cout << "update: " << rank << " " << new_data.size() << " " << owned_variable_list.size() << " "
                    << optimizer.variables_.data().size() << " ";
          for (std::size_t i = 0; i < new_data.size(); i++) {
            std::cout << owned_variable_list[i] << ":" << new_data[i] << " ";
          }
          std::cout << std::endl;

          // TODO: improve fix during refactor
          std::vector<typename T::value_type> index_map;
          for (auto id : optimizer.comm_pattern_.value().owned_variable_list) {
            index_map.push_back(optimizer.global_reduced_map_to_local_.value()[id][0]);
          }
          op::utility::accessPermuteStore(new_data, index_map, local_data);

          optimizer.variables_.data() =
              op::ReturnLocalUpdatedVariables(optimizer.comm_pattern_.value().rank_communication,
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
      grad = op::utility::parallel::concatGlobalVector(static_cast<std::size_t>(optimizer.owned_offsets_.back()),
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
}

}  // namespace op
