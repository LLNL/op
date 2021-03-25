#ifndef OP
#define OP

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

  /// Utility methods to facilitate common operations
  namespace utility {
    
    /// MPI related type traits
    namespace detail {
      template <typename T>
      struct mpi_t {
	 static constexpr MPI_Datatype type = MPI_INT;
      };

      template <>
      struct mpi_t<double> {
	static constexpr MPI_Datatype type = MPI_DOUBLE;
      };

      template <>
      struct mpi_t<int> {
	static constexpr MPI_Datatype type = MPI_INT;
      };      
    }
   
    /// Get number of variables on each rank
    template <typename T, typename V>
    std::tuple<int, std::vector<T>> gatherVariablesPerRank(op::Vector<V> & local_vector,
							   bool gatherAll = true,
							   int root = 0,
							   MPI_Comm comm = MPI_COMM_WORLD)
    {
      T local_size = local_vector.data().size();

      int nranks;
      MPI_Comm_size(comm, &nranks);
      std::vector<T> size_on_rank(nranks);
      std::vector<int> ones(nranks, 1);
      std::vector<int> offsets(nranks);
      std::iota(offsets.begin(), offsets.end(), 0);
      if (gatherAll) {
	MPI_Allgatherv(&local_size, 1, detail::mpi_t<T>::type,
		       size_on_rank.data(), ones.data(), offsets.data(),
		       detail::mpi_t<T>::type, comm);
      } else {
	MPI_Gatherv(&local_size, 1, detail::mpi_t<T>::type,
		    size_on_rank.data(), ones.data(), offsets.data(),
		    detail::mpi_t<T>::type, root, comm);
      }

      T global_size = 0;
      for (auto lsize : size_on_rank) {
	  global_size += lsize;
	}
      return std::make_tuple(global_size, size_on_rank);
    }   

    template <typename T>
    std::vector<T> buildInclusiveOffsets(std::vector<T> & values_per_rank)
    {
      std::vector<T> inclusive_offsets(values_per_rank.size()+1);
      T offset = 0;
      std::transform( values_per_rank.begin(), values_per_rank.end(),
		      inclusive_offsets.begin()+1,
		      [&](T & value) {
			return offset += value;
		      });
      return inclusive_offsets;
    }
    
    /// Assemble the gather a vector by concatination across ranks
    template <typename V>
    V concatGlobalVector(typename V::size_type global_size,			   
			 std::vector<int> & variables_per_rank,
			 std::vector<int> & offsets,
			 V & local_vector,
			 int root = 0,
			 MPI_Comm comm = MPI_COMM_WORLD)
    {
      V global_vector(global_size);

      MPI_Gatherv(local_vector.data(), local_vector.size(),
		  detail::mpi_t<typename V::value_type>::type,
		  global_vector.data(), variables_per_rank.data(), offsets.data(),
		  detail::mpi_t<typename V::value_type>::type, root, comm);
      return global_vector;
    }

    /// Assemble the gather a vector by concatination across ranks
    template <typename V>
    V concatGlobalVector(typename V::size_type global_size,
			   std::vector<int> & variables_per_rank,
			   V & local_vector,
			   int root = 0,
			   MPI_Comm comm = MPI_COMM_WORLD)
    {
      V global_vector(global_size);

      // build offsets
      auto offsets = buildInclusiveOffsets(variables_per_rank);
      return concatGlobalVector(global_size, variables_per_rank, offsets, local_vector, root, comm);
    }

    
    template <typename T>
    int Allreduce(T * local, T * global, int size, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
      return MPI_Allreduce(local, global, size, detail::mpi_t<T>::type, op, comm);
    }


    /**
     *@brief Inserts values of T and re-indexes according to M, result[M[i]] = T[i]
     *
     * Requirements:
     * range(M) <= size(R) <= size(T)
     *
     * T = w x y z a b
     * M = 3 1 2 0 5
     * R = z x y w * b 
     *
     * Re-applying the mapping T[M[i]] = R[i]
     * T2 = w x y z * *
     *
     * @param[in] vector values to permute
     * @param[in] map indices of vector for a given element result[i]
     * @param[in, out] results a vector for the results
     */
    template <typename T, typename M>
    void insertedIndexMap(T & vector, M & map, T & results) {
      assert(results.size() <= vector.size());
      assert(*std::max_element(map.begin(), map.end()) <= results.size());
      for (typename T::size_type i = 0; i < vector.size(); i++) {
	results[map[i]] = vector[i];
      }
    }

    
    /**
     *@brief Inserts values of T and re-indexes according to M, result[M[i]] = T[i]
     *
     * Requirements:
     * range(M) <= size(R) <= size(T)
     *
     * T = w x y z a b
     * M = 3 1 2 0 5
     * R = z x y w * b 
     *
     * Re-applying the mapping T[M[i]] = R[i]
     * T2 = w x y z * *
     *
     * @param[in] vector values to permute
     * @param[in] map indices of vector for a given element result[i]
     * @param[in] pad_value default padding value in result
     * @paran[in] arg_size A manually-specified size(R), otherwise size(T)
     */
    template <typename T, typename M>
    T insertedIndexMap(T & vector, M & map, typename T::value_type pad_value,
		     std::optional<typename T::size_type> arg_size = std::nullopt) {
      // if arg_size is specified we'll use that.. otherwise we'll use T
      typename T::size_type results_size = arg_size ? *arg_size : vector.size();
      assert(results_size <= vector.size());
      assert(*std::max_element(map.begin(), map.end()) <= results_size);
      T results(results_size, pad_value);
      insertedIndexMap(vector, map, results);    
      return results;
    }    

    /**
     *@brief Selects values from T and re-indexes accordint to M,  result[i] = T[M[i]]
     *
     * Requirements: size(R) = size(M), range(M) <= size(T)
     *
     * T = w x y z a b
     * M = 3 1 2 0 5
     * result = z x y w b
     *
     * M = 3 1 2 0 4
     * T1 = w x y z b
     *
     */
    template <typename T, typename M>
    T selectIndexMap(T & vector, M & map) {
      assert(*std::max_element(map.begin(), map.end()) <= vector.size());
      assert(map.size() <= vector.size());
      T result(map.size());
      for (typename T::size_type i = 0; i < vector.size(); i++) {
	result[map[i]] = vector[i];
      }
      return result;
    }    

    /**
     * @brief MPI_Scatterv on std::collections
     *
     * @param[in] sendbuff the buffer to send
     * @param[in] variables_per_rank the numbers of variables each rank, i, will recieve
     * @param[in] offsets, the exclusive scan of varaibles_per_rank
     * @param[in] recvbuff the recieve buffer with the proper size
     *
     */
    template <typename T>
    int Scatterv(T& sendbuf, std::vector<int> & variables_per_rank,
		 std::vector<int> & offsets,
                 T & recvbuff,
                 int root = 0, MPI_Comm comm = MPI_COMM_WORLD)
    {
      // only check the size of the recv buff in debug mode
      assert(static_cast<typename T::size_type>([&]() {
	  int rank;
	  MPI_Comm_rank(comm, &rank);
	  return variables_per_rank[rank];
	  }()) == recvbuff.size());
      
      return MPI_Scatterv(sendbuf.data(), variables_per_rank.data(), offsets.data(),
			  detail::mpi_t<typename T::value_type>::type,
			  recvbuff.data(), recvbuff.size(),
			  detail::mpi_t<typename T::value_type>::type, root, comm);
    }
  }
  
}



#endif
