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
  
  namespace Variables {

  // Utility class for "converting" between Variables and something else
    template <class Variables, class FieldType>
    class VariableMap {
    public:
      using ToTypeFn = std::function<FieldType(Variables &)>;
      using FromTypeFn = std::function<Variables(FieldType &)>;
    
      VariableMap(ToTypeFn to_fn, FromTypeFn from_fn) :
	toType_(to_fn), fromType_(from_fn) {}

      FieldType convertFromVariable(Variables & v) { return toType_(v); }
      Variables convertToVariable(FieldType & f) { return fromType(f); }

    protected:
      ToTypeFn toType_;
      FromTypeFn fromType_;
    };
  }    
    
  /**
   * @brief Abstracted OptimizationVector container
   *
   * The intention is for the container to be a placeholder to define general operations
   * commonly used in serial/parallel implementations on optimization variables.
   * ROL-esque ROL::Vector
   */
  template <class VectorType>
  class Vector {
  public:
    using ScatterFn = std::function<VectorType()>;
    using GatherFn = std::function<VectorType()>;
    using BoundsFn = std::function<VectorType()>;
      
    Vector(VectorType &data, BoundsFn lowerBounds, BoundsFn upperBounds) :
      lowerBounds_(lowerBounds), upperBounds_(upperBounds), data_(data) {}

    virtual VectorType gather()  { return data_;};
    virtual VectorType scatter() { return data_;};

    VectorType & data() {return data_; }
    VectorType lowerBounds() {return lowerBounds_();} 
    VectorType upperBounds() {return upperBounds_();}    
      
  protected:
    BoundsFn lowerBounds_;
    BoundsFn upperBounds_;
    VectorType & data_;
  };

  /// Abstracted Objective class
  class Objective {
  public:
    using ResultType = double;
    using SensitivityType = std::vector<double>;
    using EvalObjectiveFn =  std::function<ResultType(const std::vector<double> &)>;
    using EvalObjectiveGradFn = std::function<SensitivityType(const std::vector<double>&)>;

    /**
     * @brief Objective container class
     *
     * @param obj A simple function that calculates the objective
     * @param grad A simple function that calculates the sensitivity
     */    
    Objective(EvalObjectiveFn obj, EvalObjectiveGradFn grad,
	      double lb = -std::numeric_limits<double>::max(),
	      double ub = std::numeric_limits<double>::max()) :
      lower_bound(lb), upper_bound(ub), obj_(obj), grad_(grad)
    {}

    // return the objective evaluation
    ResultType Eval(const std::vector<double> & v) 
    {
      return obj_(v);
    }

    // return the objective gradient evaluation
    SensitivityType EvalGradient(const std::vector<double> & v) {
      return grad_(v);
    }

    double lower_bound;
    double upper_bound;
    
  protected:
    EvalObjectiveFn obj_;
    EvalObjectiveGradFn grad_;
  };

  
  // Abstracted Optimizer implementation
  class Optimizer {
  public:

    /// Ctor has deferred initialization
    explicit Optimizer () :
      go([](){}),
      update([](){}),
      iterate([](){}),
      save([](){}),
      final_obj(std::numeric_limits<double>::max())
    {  }

    /* the following methods are needed for different optimizers */
    virtual void setObjective(Objective &o) = 0;

    virtual void addConstraint(Objective &) {}
    
    /* The following methods are hooks that are different for each optimization problem */
    
    /// Start the optimization
    void Go() { go(); }
    
    /// What to do when the variables are updated
    void UpdatedVariableCallback() { update();};

    /// What to do when the solution is found. Return the objetive
    virtual double Solution() {
      return final_obj;
    }

    /// What to do at the end of an optimization iteration
    void Iteration() {
      return iterate();
    };

    /// Saves the state of the optimizer
    void SaveState() {
      return save();
    }

    /// Destructor
    virtual ~Optimizer() = default;

    // Go function to start optimization
    CallbackFn go;

    // Update callback to compute before function calculations
    CallbackFn update;

    // iterate callback to compute before
    CallbackFn iterate;

    // callback for saving current optimizer state
    CallbackFn save;
    
    // final objective value
    double final_obj;
  };
 
  /// Dynamically load an Optimizer
  template<class OptType, typename... Args>
  std::unique_ptr<OptType> PluginOptimizer(std::string optimizer_path, Args&&... args) {
    void* optimizer_plugin = dlopen(optimizer_path.c_str(), RTLD_LAZY);

    if (!optimizer_plugin)
      {
	std::cout << dlerror() << std::endl;
	return nullptr;
      }
    
    auto load_optimizer = (std::unique_ptr<OptType> (*)(Args...)) dlsym( optimizer_plugin, "load_optimizer");
    if (load_optimizer) {
      return load_optimizer(std::forward<Args>(args)...);
    } else {      
      return nullptr;
    }
  }

  /// Generate an objective function that performs a global reduction
  template <typename V, typename T>
  auto ReduceObjectiveFunction(std::function<V(const T &)> && local_func, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
    return [=](const T & variables) {
      V local_sum = local_func(variables);
      V global_sum;
      auto error = op::mpi::Allreduce(local_sum, global_sum, op, comm);
      if (error != MPI_SUCCESS) {
	std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
      }
      return global_sum;
    };  
  }

  /// Generate an objective gradient function that takes local variables and reduces them in parallel to locally "owned" variables
  template <typename T, typename I>
  auto OwnedLocalObjectiveGradientFunction(utility::RankCommunication<T> & info,
					   I & global_ids_to_local,
					   std::function<std::vector<double>(const std::vector<double> &)> local_obj_grad_func,
					   std::function<double(const std::vector<double> &)> local_reduce_func,
					   MPI_Comm comm = MPI_COMM_WORLD) {
    return [&] (const std::vector<double> & local_variables) {
      // First we send any local gradient information to the ranks that "own" the variables
      auto local_obj_gradient = local_obj_grad_func(local_variables);
      auto recv_data = op::utility::sendToOwners(info, local_obj_gradient, comm);
      auto combine_data = op::utility::remapRecvDataIncludeLocal(info.recv, recv_data, global_ids_to_local, local_obj_gradient);
      std::vector<double> reduced_local_variables =
	op::utility::reduceRecvData(combine_data, local_reduce_func);
      return reduced_local_variables;
    };
  }

  
} // namespace op

