// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

namespace op {

  // This constructor is used when variables don't overlap on different ranks
  template <typename T>
  NLopt<T>::NLopt(op::Vector<std::vector<double>>& variables, NLoptOptions& o, MPI_Comm comm, std::optional<CommPattern<T>> comm_pattern_info)
    : comm_(comm), global_variables_(0), variables_(variables), options_(o), comm_pattern_(comm_pattern_info), global_reduced_map_to_local_({}), num_local_owned_variables_(0)
  {
    std::cout << "NLOpt wrapper constructed" << std::endl;

    auto rank = op::mpi::getRank(comm_);
  
    // Since this optimizer runs in serial we need to figure out the global number of decisionvariables
    // in "advanced" mode some of the local_variables may not be unique to each partition
    if (comm_pattern_.has_value()) {
      // figure out what optimization variables we actually own
      auto & comm_pattern = comm_pattern_.value();
      num_local_owned_variables_ = comm_pattern.owned_variable_list.get().size();
      global_reduced_map_to_local_ = op::utility::inverseMap(comm_pattern.local_variable_list.get());
    } else {
      num_local_owned_variables_ = variables.data().size();
    }
    // in "simple" mode the local_variables are unique to each partition
    auto [global_size, variables_per_rank] = op::utility::parallel::gatherVariablesPerRank<int>(num_local_owned_variables_);
    owned_variables_per_rank_ = variables_per_rank;
    owned_offsets_ = op::utility::buildInclusiveOffsets(owned_variables_per_rank_);
    
    if (rank == 0) {
      global_variables_.resize(global_size);  
      nlopt_ = std::make_unique<nlopt::opt>(nlopt::LD_MMA, global_variables_.size());  
    }
  
    // Set variable bounds
    auto lowerBounds = variables.lowerBounds();
    auto upperBounds = variables.upperBounds();

    // Adjust in "advanced" mode
    if (comm_pattern_.has_value()) {
      auto & reduced_variable_list = comm_pattern_.value().owned_variable_list.get();
      lowerBounds = op::utility::permuteMapAccessStore(lowerBounds, reduced_variable_list, global_reduced_map_to_local_.value());
      upperBounds = op::utility::permuteMapAccessStore(upperBounds, reduced_variable_list, global_reduced_map_to_local_.value());
    }
    
    auto global_lower_bounds = op::utility::parallel::concatGlobalVector(global_size, owned_variables_per_rank_, owned_offsets_, lowerBounds, false); // gather on rank 0
  
    auto global_upper_bounds = op::utility::parallel::concatGlobalVector(global_size, owned_variables_per_rank_, owned_offsets_, upperBounds, false); // gather on rank 0

    // save initial set of variables to detect if variables changed
    if (comm_pattern_.has_value()) {
      auto & reduced_variable_list = comm_pattern_.value().owned_variable_list.get();     
      auto reduced_previous_local_variables = op::utility::permuteMapAccessStore(variables.data(), reduced_variable_list, global_reduced_map_to_local_.value());
      previous_variables_ = op::utility::parallel::concatGlobalVector(global_size, owned_variables_per_rank_, owned_offsets_, reduced_previous_local_variables, false); // gather on rank 0
      
    } else {
      previous_variables_ = op::utility::parallel::concatGlobalVector(global_size, owned_variables_per_rank_, owned_offsets_, variables.data(), false); // gather on rank 0
    }

    // initialize our starting global variables
    global_variables_ = previous_variables_;
  
    if (rank == 0) {

      nlopt_->set_lower_bounds(global_lower_bounds);
      nlopt_->set_upper_bounds(global_upper_bounds); 

      // Optimizer settings
      // Process Integer options
      if (o.Int.find("maxeval") != o.Int.end()) nlopt_->set_maxeval(o.Int["maxeval"]);

      // Process Double options
      if (o.Double.find("xtol_rel") != o.Double.end())
	nlopt_->set_xtol_rel(o.Double["xtol_rel"]);  // various tolerance stuff ;)

      // Check if constraint_tol key exists in options.Double
      if (options_.Double.find("constraint_tol") == options_.Double.end()) {
	options_.Double["constraint_tol"] = 0.;
      }
    }  

    // Create default go
  
    if (rank == 0) {
      go = [&]() {

	nlopt_->set_min_objective(NLoptFunctional<T>, &obj_info_[0]);
	for (auto &constraint : constraints_info_) {
	  nlopt_->add_inequality_constraint(NLoptFunctional<T>, &constraint, constraint.constraint_tol);
	}
      
	nlopt_->optimize(global_variables_, final_obj);
	// propagate solution objective to all ranks
	std::vector<int> state {op::State::SOLUTION_FOUND};
	op::mpi::Broadcast(state, 0, comm_);

	std::vector<double> obj(1, final_obj);
	op::mpi::Broadcast (obj, 0, comm_);

      };
    } else {
      go = serialOptimizerNonRootWaitLoop(// update variables
					  [&, rank] () {
					    // recieve the incoming variables
					    std::vector<double> owned_data(num_local_owned_variables_);
					    std::vector<double> empty;
					    op::mpi::Scatterv(empty, owned_variables_per_rank_, owned_offsets_, owned_data, 0, comm_);
					    if (comm_pattern_.has_value()) {
					      // repropagate back to non-owning ranks
					      std::vector<double> local_data(variables_.data().size());
					      op::utility::accessPermuteStore(owned_data, comm_pattern_.value().owned_variable_list.get(), local_data);
					      std::cout << " local_data :" << rank << " ";
					      for (auto v : local_data) {
						std::cout << v << " ";
					      }
					      std::cout << std::endl;
					      
					      variables_.data() =
						op::ReturnLocalUpdatedVariables(comm_pattern_.value().rank_communication.get(), global_reduced_map_to_local_.value(), local_data);
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
					  },
					  // obj_grad
					  [&]() {
					    std::vector<double> grad(variables_.data().size());
					    // Call NLoptFunctional on non-root-rank
					    NLoptFunctional<T>(variables_.data(), grad, &obj_info_[0]);
					  },
					  // obj_Eval
					  [&]() {
					    std::vector<double> grad;
					    // Call NLoptFunctional on non-root-rank
					    NLoptFunctional<T>(variables_.data(), grad, &obj_info_[0]);		  
					  },
					  // constraint states
					  [&](int state) {
					    // just an eval routine
					    std::vector<double> grad;
					    // Call NLoptFunctional on non-root-rank
					    NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[state]);	  
					  },				
					  // constraint grad states
					  [&](int state) {
					    // this is a constraint gradient call
					    std::vector<double> grad(variables_.data().size());
					    // Call NLoptFunctional on non-root-rank
					    NLoptFunctional<T>(variables_.data(), grad, &constraints_info_[state]);
					  },
					  // Solution state
					  [&] () {
					    // The solution has been found.. recieve the objective
					    std::vector<double> obj(1);
					    op::mpi::Broadcast (obj, 0, comm_);
					    final_obj = obj[0];
					    // exit this loop finally!
					  },
					  // unknown state
					  [&](int state){
					    // this is an unknown state!
					    std::cout << "Unknown State: " << state << std::endl;
					    MPI_Abort(comm_, state);
					  },
					  constraints_info_,
					  final_obj,
					  comm_);
    }
  }

  template <typename T>
  void NLopt<T>::setObjective(op::Functional& o) {
    obj_info_.clear();
    obj_info_.emplace_back(op::detail::FunctionalInfo<T>{.obj=o, .nlopt=*this, .state=-1, .constraint_tol = 0.});
  }

  template <typename T>
  void NLopt<T>::addConstraint(op::Functional& o)
  {
    constraints_info_.emplace_back(op::detail::FunctionalInfo<T>{.obj=o, .nlopt=*this, .state=static_cast<int>(constraints_info_.size()), .constraint_tol = options_.Double["constraint_tol"]});
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
									       op::NLoptOptions&              options)
{
  return std::make_unique<op::NLopt<op::nlopt_index_type>>(variables, options);
}

