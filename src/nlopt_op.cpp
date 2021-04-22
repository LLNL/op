// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

namespace op {

  // This constructor is used when variables don't overlap on different ranks
  template <typename T>
  NLopt<T>::NLopt(op::Vector<std::vector<double>>& variables, NLoptOptions& o, MPI_Comm comm, std::optional<op::utility::RankCommunication<T>> comm_pattern)
    : comm_(comm), global_variables_(0), variables_(variables), options_(o), comm_pattern_(comm_pattern), reduced_variable_list_({}), global_reduced_map_to_local_({})
  {
    std::cout << "NLOpt wrapper constructed" << std::endl;

    auto rank = op::mpi::getRank(comm_);
  
    // Since this optimizer runs in serial we need to figure out the global number of decisionvariables
    int num_local_owned_variables;
    // in "advanced" mode some of the local_variables may not be unique to each partition
    if (comm_pattern_.has_value()) {
      // figure out what optimization variables we actually own
      auto & recv = comm_pattern_.value().recv;

      using variable_type = typename T::value_type;
      std::unordered_map<variable_type, variable_type> temp_map;
      for (auto [rank, recv_vars] : recv) {
	// we blindly insert variables into the map.. some of the variables may be part of a reduction for a variable we own.. that's fine
	for (auto v : recv_vars) {
	  temp_map[v] = v;
	}
      }
      reduced_variable_list_ = op::utility::mapToVector(temp_map);
      num_local_owned_variables = reduced_variable_list_.value().size();
      global_reduced_map_to_local_ = op::utility::inverseMap(reduced_variable_list_.value());
    } else {
      num_local_owned_variables = variables.data().size();
    }
    // in "simple" mode the local_variables are unique to each partition
    auto [global_size, variables_per_rank] = op::utility::parallel::gatherVariablesPerRank(num_local_owned_variables);
    variables_per_rank_ = variables_per_rank;
    offsets_ = op::utility::buildInclusiveOffsets(variables_per_rank_);
    
    if (rank == 0) {
      global_variables_.resize(global_size);  
      nlopt_ = std::make_unique<nlopt::opt>(nlopt::LD_MMA, global_variables_.size());  
    }
  
    // Set variable bounds
    auto lowerBounds = variables.lowerBounds();
    auto upperBounds = variables.upperBounds();

    // Adjust in "advanced" mode
    if (reduced_variable_list_.has_value()) {
      auto & reduced_variable_list = reduced_variable_list_.value();
      lowerBounds = op::utility::permuteAccessStore(lowerBounds, reduced_variable_list);
      upperBounds = op::utility::permuteAccessStore(upperBounds, reduced_variable_list);
    }
    
    auto global_lower_bounds = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, lowerBounds, false); // gather on rank 0
  
    auto global_upper_bounds = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, upperBounds, false); // gather on rank 0

    // save initial set of variables to detect if variables changed
    if (reduced_variable_list_.has_value()) {
      auto & reduced_variable_list = reduced_variable_list_.value();
      auto reduced_previous_local_variables = op::utility::permuteAccessStore(variables.data(), reduced_variable_list);
      previous_variables_ = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, reduced_previous_local_variables, false); // gather on rank 0
      
    } else {
      previous_variables_ = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, variables.data(), false); // gather on rank 0
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
      };
    } else {
      go = serialOptimizerNonRootWaitLoop(// state machine states
					  std::unordered_map<op::State, std::function<void()>>{
					    {op::State::UPDATE_VARIABLES ,
						[&] () {
						// recieve the incoming variables
						std::vector<double> new_data(reduced_variable_list_.has_value() ? reduced_variable_list_.value().size() : variables_.data().size());
						std::vector<double> empty;
						op::mpi::Scatterv(empty, variables_per_rank_, offsets_, new_data, 0, comm_);
						if (comm_pattern_.has_value()) {
						// repropagate back to non-owning ranks
						variables_.data() =
						  op::ReturnLocalUpdatedVariables(comm_pattern_.value(), global_reduced_map_to_local_.value(), new_data);
						} else {
						  variables_.data() = new_data;
						}
						
						// Call update
						UpdatedVariableCallback();
					      }},
					      {op::State::OBJ_GRAD,
						  [&]() {
						  std::vector<double> grad(variables_.data().size());
						  // Call NLoptFunctional on non-root-rank
						  NLoptFunctional<T>(variables_.data(), grad, &obj_info_);
						}},
						{op::State::OBJ_EVAL,
						    [&]() {
						    std::vector<double> grad;
						    // Call NLoptFunctional on non-root-rank
						    NLoptFunctional<T>(variables_.data(), grad, &obj_info_);		  
						  }}},
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
					    std::vector<double> obj;
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
					  constraints_info_.size(),
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

