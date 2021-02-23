#ifndef OP
#define OP

#include <functional>
#include <vector>
#include <memory>

/// Namespace for the OP interface
namespace op {

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
      
    Vector(BoundsFn lowerBounds, BoundsFn upperBounds, GatherFn gather, ScatterFn scatter) :
      scatter_(scatter), gather_(gather), lowerBounds_(lowerBounds), upperBounds_(upperBounds) {}

    VectorType gather() {return gather_();}
    VectorType scatter() {return scatter_(); }
      
    VectorType lowerBounds() {return lowerBounds_();} 
    VectorType upperBounds() {return upperBounds_();}
      
  protected:
    ScatterFn scatter_;
    GatherFn gather_;
    BoundsFn lowerBounds_;
    BoundsFn upperBounds_;
  };
  
  // Abstracted Optimizer implementation
  class Optimizer {
  public:
    
    Optimizer (CallbackFn setup) 
    {
      setup();
    }
    
    /// What to do when the variables are updated
    virtual void UpdatedVariableCallback() = 0;

    /// What to do when the solution is found
    virtual void SolutionCallback() {};

    /// What to do at the end of an optimization iteration
    virtual void IterationCallback() {};

    /// Saves the state of the optimizer
    virtual void SaveState() {}

    /// Destructor
    virtual ~Optimizer() = default;
  };

  extern "C" std::unique_ptr<Optimizer> load_optimizer(CallbackFn setup);
  
  /// Abstracted Objective class
  template <typename ResultType, class SensitivityType>
  class Objective {
  public:
    using EvalObjectiveFn =  std::function<ResultType()>;
    using EvalObjectiveGradFn = std::function<SensitivityType()>;

    /**
     * @brief Objective container class
     *
     * @param obj A simple function that calculates the objective
     * @param grad A simple function that calculates the sensitivity
     */    
    Objective(EvalObjectiveFn obj, EvalObjectiveGradFn grad) :
      obj_(obj), grad_(grad) {}

    // return the objective evaluation
    ResultType getObjective() 
    {
      return obj_();
    }

    // return the objective gradient evaluation
    SensitivityType getObjectiveGradient() {
      return grad_();
    }
    
  protected:
    EvalObjectiveFn obj_;
    EvalObjectiveGradFn grad_;
  };

  /// Dynamically load an Optimizer
  extern std::unique_ptr<Optimizer> PluginOptimizer(std::string optimizer_path, CallbackFn setup);
  
}



#endif
