#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <dlfcn.h>
#include <iostream>
#include <mpi.h>
#include <tuple>
#include <numeric>
#include <cassert>
#include <optional>
#include <algorithm>
#include <set>
#include "op_utility.hpp"

/// Namespace for the OP interface
namespace op {

/// Callback function type
using CallbackFn = std::function<void()>;

/** Go Functor
    A functor to hold the optimization.Go() and .Preprocess() functions
 */
class Go {
public:
  /// Define preprocess action
  Go& onPreprocess(const CallbackFn& preprocess)
  {
    preprocess_ = preprocess;
    return *this;
  }

  /// Define Go action
  Go& onGo(const CallbackFn& go)
  {
    go_ = go;
    return *this;
  }

  // Define the operator
  void operator()()
  {
    preprocess_();
    go_();
  }

protected:
  CallbackFn go_;
  CallbackFn preprocess_;
};

namespace Variables {

/// Utility class for "converting" between Variables and something else
template <class Variables, class FieldType>
class VariableMap {
public:
  using ToTypeFn   = std::function<FieldType(Variables&)>;
  using FromTypeFn = std::function<Variables(FieldType&)>;

  VariableMap(ToTypeFn to_fn, FromTypeFn from_fn) : toType_(to_fn), fromType_(from_fn) {}

  FieldType convertFromVariable(Variables& v) { return toType_(v); }
  Variables convertToVariable(FieldType& f) { return fromType(f); }

protected:
  ToTypeFn   toType_;
  FromTypeFn fromType_;
};
}  // namespace Variables

/**
 * @brief Abstracted Optimization Vector container
 *
 * The intention is for the container to act as a general abstraction for optimization variables
 */
template <class VectorType>
class Vector {
public:
  using ScatterFn = std::function<VectorType()>;
  using GatherFn  = std::function<VectorType()>;
  using BoundsFn  = std::function<VectorType()>;

  Vector(VectorType& data, BoundsFn lowerBounds, BoundsFn upperBounds)
      : lowerBounds_(lowerBounds),
        upperBounds_(upperBounds),
        data_(data),
        gather([&]() { return data; }),
        scatter([&]() { return data; })
  {
  }

  /// Get the underlying data
  VectorType& data() { return data_; }

  /// Get the lower bounds for each local optimization variable
  VectorType lowerBounds() { return lowerBounds_(); }

  /// Get the upper bounds for each local optimization variable
  VectorType upperBounds() { return upperBounds_(); }

protected:
  BoundsFn    lowerBounds_;
  BoundsFn    upperBounds_;
  VectorType& data_;

public:
  /// Gather data function
  GatherFn gather;

  /// Scatter data function
  ScatterFn scatter;
};

/// Abstracted Objective Functional class
class Functional {
public:
  using ResultType          = double;
  using SensitivityType     = std::vector<double>;
  using EvalObjectiveFn     = std::function<ResultType(const std::vector<double>&)>;
  using EvalObjectiveGradFn = std::function<SensitivityType(const std::vector<double>&)>;

  static constexpr double default_min = -std::numeric_limits<double>::max();
  static constexpr double default_max = std::numeric_limits<double>::max();

  /**
   * @brief Objective container class
   *
   * @param obj A simple function that calculates the objective
   * @param grad A simple function that calculates the sensitivity
   */
  Functional(EvalObjectiveFn obj, EvalObjectiveGradFn grad, double lb = default_min, double ub = default_max)
      : lower_bound(lb), upper_bound(ub), obj_(obj), grad_(grad)
  {
  }

  /**
   * @brief  Return the objective evaluation
   *
   * @param[in] v input optimization vector to evaluate
   */
  ResultType Eval(const std::vector<double>& v) { return obj_(v); }

  /**
   * @brief return the objective gradient evaluation
   *
   * @param[in] v input optimization vector to evaluate
   */
  SensitivityType EvalGradient(const std::vector<double>& v) { return grad_(v); }

  /// Lower bounds for this optimization functional
  double lower_bound;

  /// Upper bounds for this optimization functional
  double upper_bound;

protected:
  EvalObjectiveFn     obj_;
  EvalObjectiveGradFn grad_;
};

/// Abstracted Optimizer implementation
class Optimizer {
public:
  /// Ctor has deferred initialization
  explicit Optimizer() : update([]() {}), iterate([]() {}), save([]() {}), final_obj(std::numeric_limits<double>::max())
  {
  }

  /* the following methods are needed for different optimizers */

  /**
   * @brief Sets the optimization objective
   *
   * @param[in] o Objective Functional
   */
  virtual void setObjective(Functional& o) = 0;

  /**
   * @brief Adds a constraint for the optimization problem
   *
   * @param[in] o Constraint Functional
   */
  virtual void addConstraint(Functional&) {}

  /* The following methods are hooks that are different for each optimization problem */

  /// Start the optimization
  void Go() { go(); }

  /// What to do when the variables are updated
  void UpdatedVariableCallback() { update(); };

  /// What to do when the solution is found. Return the objetive
  virtual double Solution() { return final_obj; }

  /// What to do at the end of an optimization iteration
  void Iteration() { return iterate(); };

  /// Saves the state of the optimizer
  void SaveState() { return save(); }

  /// Destructor
  virtual ~Optimizer() = default;

  /// Go function to start optimization
  op::Go go;

  /// Update callback to compute before function calculations
  CallbackFn update;

  /// iterate callback to compute before
  CallbackFn iterate;

  /// callback for saving current optimizer state
  CallbackFn save;

  /// final objective value
  double final_obj;
};

/**
 * @brief  Dynamically load an Optimizer
 *
 * @param[in] optimizer_path path to dynamically loadable .so plugin
 * @param[in] args A list of args to pass in for initialization
 */
template <class OptType, typename... Args>
std::unique_ptr<OptType> PluginOptimizer(std::string optimizer_path, Args&&... args)
{
  void* optimizer_plugin = dlopen(optimizer_path.c_str(), RTLD_LAZY);

  if (!optimizer_plugin) {
    std::cout << dlerror() << std::endl;
    return nullptr;
  }

  auto load_optimizer =
      reinterpret_cast<std::unique_ptr<OptType> (*)(Args...)>(dlsym(optimizer_plugin, "load_optimizer"));
  if (load_optimizer) {
    return load_optimizer(std::forward<Args>(args)...);
  } else {
    return nullptr;
  }
}

/**
 *@brief  Generate an objective function that performs a global reduction
 *
 * @param[in] local_func A user-defined function to compute a rank-local objective-contribution
 * @param[in] op The MPI reduction operation
 * @param[in] comm The MPI communicator
 */
template <typename V, typename T>
auto ReduceObjectiveFunction(const std::function<V(T)>& local_func, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD)
{
  return [=](T variables) {
    V    local_sum = local_func(variables);
    V    global_sum;
    auto error = op::mpi::Allreduce(local_sum, global_sum, op, comm);
    if (error != MPI_SUCCESS) {
      std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
    }
    return global_sum;
  };
}

/**
 * @brief Generate an objective gradient function that takes local variables and reduces them in parallel to locally
 * "owned" variables
 *
 *
 * @param[in] info RankCommunication struct for local_variables
 * @param[in] global_ids_to_local A vector mapping of global ids corresponding to local_variable indices
 * @param[in] local_obj_grad_func The rank-local gradient contributions corresponding to local_variables
 * @param[in] local_reduce_func A serial user-defined function computed on "owned" variables over both recieved
 * contributions from other ranks and rank-local gradient contributions
 * @param[in] comm the MPI communicator
 */
template <typename T, typename I>
auto OwnedLocalObjectiveGradientFunction(
    utility::RankCommunication<T>& info, I& global_ids_to_local, T& reduced_id_list,
    std::function<std::vector<double>(const std::vector<double>&)> local_obj_grad_func,
    std::function<double(const std::vector<double>&)> local_reduce_func, MPI_Comm comm = MPI_COMM_WORLD)
{
  return [=, &info, &global_ids_to_local](const std::vector<double>& local_variables) {
    // First we send any local gradient information to the ranks that "own" the variables
    auto local_obj_gradient = local_obj_grad_func(local_variables);
    auto recv_data          = op::utility::parallel::sendToOwners(info, local_obj_gradient, comm);
    auto combine_data =
        op::utility::remapRecvDataIncludeLocal(info.recv, recv_data, global_ids_to_local, local_obj_gradient);
    std::vector<double> reduced_local_variables = op::utility::reduceRecvData(combine_data, local_reduce_func);
    // At this point the data should be reduced but it's still in the local-data view
    return op::utility::permuteMapAccessStore(reduced_local_variables, reduced_id_list, global_ids_to_local);
  };
}

/**
 * @brief Generate update method to propagate owned local variables back to local variables in parallel
 *
 * for the variables a rank owns.. update() should propagate those
 * for the variables an update does not own.. they will be in returned_data
 * returned_remapped_data is a map[local_ids] -> values
 * we want to write it back into our local variable array
 *
 * @param[in] info The RankCommunication information corresponding to local_variable data
 * @param[in] global_ids_to_local A vector mapping of global ids corresponding to local_variable indices
 * @param[in] reduced_values The rank-local "owned" variables
 */
template <typename T, typename I, typename ValuesType>
ValuesType ReturnLocalUpdatedVariables(utility::RankCommunication<T>& info, I& global_ids_to_local,
                                       ValuesType& reduced_values)
{
  auto returned_data = op::utility::parallel::returnToSender(info, reduced_values);
  auto returned_remapped_data =
      op::utility::remapRecvDataIncludeLocal(info.send, returned_data, global_ids_to_local, reduced_values);
  ValuesType updated_local_variables;
  if (info.send.size() == 0) {
    // we own all the variables
    updated_local_variables = reduced_values;
  } else {
    updated_local_variables = op::utility::reduceRecvData(
        returned_remapped_data,
        op::utility::reductions::firstOfCollection<typename decltype(returned_remapped_data)::mapped_type>);
  }
  return updated_local_variables;
}

/**
   AdvancedRegistration procedure given


@return CommPattern to use with op::Optimizer
 */
template <typename T>
auto AdvancedRegistration(T& global_ids_on_rank, int root = 0, MPI_Comm mpicomm = MPI_COMM_WORLD)
{

  // check if global_ids_on_rank are unique
  std::set<typename T::value_type> label_set(global_ids_on_rank.begin(), global_ids_on_rank.end());
  assert(label_set.size() == global_ids_on_rank.size());
  
  auto [global_size, variables_per_rank] =
      op::utility::parallel::gatherVariablesPerRank<int>(global_ids_on_rank.size(), true, root, mpicomm);
  auto offsets              = op::utility::buildInclusiveOffsets(variables_per_rank);
  auto all_global_ids_array = op::utility::parallel::concatGlobalVector(global_size, variables_per_rank,
                                                                        global_ids_on_rank, true, root, mpicomm);

  auto global_local_map = op::utility::inverseMap(global_ids_on_rank);
  auto recv_send_info =
      op::utility::parallel::generateSendRecievePerRank(global_local_map, all_global_ids_array, offsets, mpicomm);
  auto reduced_dvs_on_rank = op::utility::filterOut(global_ids_on_rank, recv_send_info.send);

  return op::utility::CommPattern{recv_send_info, reduced_dvs_on_rank, global_ids_on_rank};
}

}  // namespace op
