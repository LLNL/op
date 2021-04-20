// NLopt op::Optimizer implementation

#include "nlopt_op.hpp"
#include <iostream>

namespace op {

  // This constructor is used when variables don't overlap on different ranks
  NLopt::NLopt(op::Vector<std::vector<double>>& variables, Options& o, MPI_Comm comm)
    : comm_(comm), global_variables_(0), variables_(variables), options_(o)
{
  std::cout << "NLOpt wrapper constructed" << std::endl;

  auto rank = op::mpi::getRank(comm_);
  
  // Since this optimizer runs in serial we need to figure out the global number of decisionvariables
  auto [global_size, variables_per_rank] = op::utility::parallel::gatherVariablesPerRank<int>(variables.data().size());
  variables_per_rank_ = variables_per_rank;
  offsets_ = op::utility::buildInclusiveOffsets(variables_per_rank_);
  if (rank == 0) {
    global_variables_.resize(global_size);  
    nlopt_ = std::make_unique<nlopt::opt>(nlopt::LD_MMA, global_variables_.size());  
  }
  
  // Set variable bounds
  auto lowerBounds = variables.lowerBounds();
  auto global_lower_bounds = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, lowerBounds, false); // gather on rank 0
  
  auto upperBounds = variables.upperBounds();
  auto global_upper_bounds = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, upperBounds, false); // gather on rank 0

  // save initial set of variables to detect if variables changed
  previous_variables_ = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank_, offsets_, variables.data(), false); // gather on rank 0

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

      nlopt_->optimize(global_variables_, final_obj);
      // propagate solution objective to all ranks
      std::vector<int> state {op::NLopt::State::SOLUTION_FOUND};
      op::mpi::Broadcast(state, 0, comm_);      
    };
  } else {
    go = [&]() {
      // non-root ranks will be in a perpetual wait loop until we find a solution
      while (final_obj == std::numeric_limits<double>::max()) {
	// set up to recieve
	std::vector<int> state(1);
	op::mpi::Broadcast(state, 0, comm_);

	if (state[0] == op::NLopt::State::UPDATE_VARIABLES) {
	  // recieve the incoming variables
	  std::vector<double> empty;
	  op::mpi::Scatterv(empty, variables_per_rank_, offsets_, variables_.data(), 0, comm_);
	  // Call update
	  UpdatedVariableCallback();
	} else if (state[0] == op::NLopt::State::SOLUTION_FOUND) {
	  // The solution has been found.. recieve the objective
	  std::vector<double> obj;
	  op::mpi::Broadcast (obj, 0, comm_);
	  final_obj = obj[0];
	  // exit this loop finally!
	  break;
	} else if (state[0] == op::NLopt::State::OBJ_GRAD) {
	  std::vector<double> grad(variables_.data().size());
	  // Call NLoptFunctional on non-root-rank
	  NLoptFunctional(variables_.data(), grad, &obj_info_);
	} else if (state[0] == op::NLopt::State::OBJ_EVAL) {
	  std::vector<double> grad;
	  // Call NLoptFunctional on non-root-rank
	  NLoptFunctional(variables_.data(), grad, &obj_info_);		  
	} else if (state[0] >= static_cast<int>(constraints_info_.size()) && state[0] < static_cast<int>(constraints_info_.size()) * 2) {
	  // this is a constraint gradient call
	  std::vector<double> grad(variables_.data().size());
	  // Call NLoptFunctional on non-root-rank
	  NLoptFunctional(variables_.data(), grad, &constraints_info_[state[0] % constraints_info_.size()]);
	} else if (state[0] >= 0 && state[0] < static_cast<int>(constraints_info_.size())) {
	  // just an eval routine
	  std::vector<double> grad;
	  // Call NLoptFunctional on non-root-rank
	  NLoptFunctional(variables_.data(), grad, &constraints_info_[state[0] % static_cast<int>(constraints_info_.size())]);	  
	} else {
	  // this is an unknown state!
	  std::cout << "Unknown State: " << state[0] << std::endl;
	  MPI_Abort(comm_, state[0]);
	  break;
	}
	
      }
    };
  }
}

void NLopt::setObjective(op::Functional& o) {
  obj_info_.clear();
  obj_info_.emplace_back(FunctionalInfo{.obj=o, .nlopt=*this, .state=-1});
  nlopt_->set_min_objective(NLoptFunctional, &obj_info_[0]); }

void NLopt::addConstraint(op::Functional& o)
{
  constraints_info_.emplace_back(FunctionalInfo{.obj=o, .nlopt=*this, .state=static_cast<int>(constraints_info_.size())});
  nlopt_->add_inequality_constraint(NLoptFunctional,
				    &constraints_info_.back(),
				    options_.Double["constraint_tol"]);
};

}  // namespace op
// end NLopt implementation

/**
 * @brief nlopt plugin loading implementation
 *
 * @param[in] variables Optimization variable abstraction
 * @param[in] options op::NLopt option struct
 */
extern "C" std::unique_ptr<op::NLopt> load_optimizer(op::Vector<std::vector<double>>& variables,
                                                     op::NLopt::Options&              options)
{
  return std::make_unique<op::NLopt>(variables, options);
}

