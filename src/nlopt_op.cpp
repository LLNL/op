// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

namespace op {

// This constructor is used when variables don't overlap on different ranks
template <typename T>
NLopt<T>::NLopt(op::Vector<std::vector<double>>& variables, NLoptOptions& o, std::optional<MPI_Comm> comm,
                std::optional<op::utility::CommPattern<T>> comm_pattern_info)
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
      global_variables_.resize(global_size);
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
  if (comm_pattern_.has_value()) {
    auto& reduced_variable_list = comm_pattern_.value().owned_variable_list;
    lowerBounds =
        op::utility::permuteMapAccessStore(lowerBounds, reduced_variable_list, global_reduced_map_to_local_.value());
    upperBounds =
        op::utility::permuteMapAccessStore(upperBounds, reduced_variable_list, global_reduced_map_to_local_.value());
  }

  // save initial set of variables to detect if variables changed
  // set previous_variables to make the check
  if (comm_pattern_.has_value()) {
    auto& reduced_variable_list            = comm_pattern_.value().owned_variable_list;
    auto  reduced_previous_local_variables = op::utility::permuteMapAccessStore(variables.data(), reduced_variable_list,
                                                                               global_reduced_map_to_local_.value());
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
            op::utility::accessPermuteStore(owned_data, comm_pattern_.value().owned_variable_list, local_data);
            std::cout << " local_data :" << rank << " ";
            for (auto v : local_data) {
              std::cout << v << " ";
            }
            std::cout << std::endl;

            variables_.data() = op::ReturnLocalUpdatedVariables(comm_pattern_.value().rank_communication,
                                                                global_reduced_map_to_local_.value(), local_data);
            std::cout << " variables.data :" << rank << " ";
            for (auto v : variables_.data()) {
              std::cout << v << " ";
            }
            std::cout << std::endl;

          } else {
            variables_.data() = owned_data;
          }

          std::cout << " rank:" << rank << " ";
          for (auto v : variables_.data()) {
            std::cout << v << " ";
          }
          std::cout << std::endl;
          // Call update
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
              NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[state]);
            })
        .onConstraintsGrad(
            // constraint grad states
            [&](int state) {
              // this is a constraint gradient call
              std::vector<double> grad(variables_.data().size());
              // Call NLoptFunctional on non-root-rank
              NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[state]);
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

template <typename T>
void NLopt<T>::setObjective(op::Functional& o)
{
  obj_info_.clear();
  obj_info_.emplace_back(
      op::detail::FunctionalInfo<T>{.obj = o, .nlopt = *this, .state = State::OBJ_EVAL, .constraint_tol = 0.});
}

template <typename T>
void NLopt<T>::addConstraint(op::Functional& o)
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
};

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
                                         ? optimizer.comm_pattern_.value().owned_variable_list.size()
                                         : optimizer.variables_.data().size());
        op::mpi::Scatterv(x_temp, optimizer.owned_variables_per_rank_, optimizer.owned_offsets_, new_data, 0,
                          optimizer.comm_);

        if (optimizer.comm_pattern_.has_value()) {
          // repropagate back to non-owning ranks
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

}  // namespace op
// end NLopt implementation

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
