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
#include <algorithm>
#include <set>

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

      template <>
      struct mpi_t<unsigned long> {
	static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG;
      };
    }

    // Get rank
    int getRank(MPI_Comm comm = MPI_COMM_WORLD) {
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }
    
    /// Get number of variables on each rank
    template <typename T>
    auto gatherVariablesPerRank(T local_vector_size,
				bool gatherAll = true,
				int root = 0,
				MPI_Comm comm = MPI_COMM_WORLD)
    {
      T local_size = local_vector_size;

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
     *@brief Retrieves from T and stores in permuted mapping M, result[M[i]] = T[i]
     *
     * T is not guarnateed to work in-place.
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
    void accessPermuteStore(T & vector, M & map, T & results) {
      assert(results.size() <= vector.size());
      assert(static_cast<typename T::size_type>(*std::max_element(map.begin(), map.end())) <= results.size());
      for (typename T::size_type i = 0; i < vector.size(); i++) {
	results[map[i]] = vector[i];
      }
    }

    
    /**
     *@brief Retrieves from T in order and stores in permuted mapping M, result[M[i]] = T[i]
     * 
     * T is not guarnateed to work in-place. This method returns results in a newly padded vector
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
    T accessPermuteStore(T & vector, M & map, typename T::value_type pad_value,
		     std::optional<typename T::size_type> arg_size = std::nullopt) {
      // if arg_size is specified we'll use that.. otherwise we'll use T
      typename T::size_type results_size = arg_size ? *arg_size : vector.size();
      assert(results_size <= vector.size());
      assert(static_cast<typename T::size_type>(*std::max_element(map.begin(), map.end())) <= results_size);
      T results(results_size, pad_value);
      accessPermuteStore(vector, map, results);    
      return results;
    }    

    /**
     *@brief Retrieves from T using a permuted mapping M and stores in order,  result[i] = T[M[i]]
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
    T permuteAccessStore(T & vector, M & map) {
      assert(static_cast<typename T::size_type>(*std::max_element(map.begin(), map.end())) <= vector.size());
      assert(map.size() <= vector.size());
      T result(map.size());
      for (typename T::size_type i = 0; i < vector.size(); i++) {
	result[i] = vector[map[i]];
      }
      return result;
    }    

    /// MPI_Scatter a vector to all ranks on the communicator
    template <typename T>
    int Broadcast(T & buf, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      return MPI_Bcast(buf.data(),
		       buf.size(),
		       detail::mpi_t<typename T::value_type>::type,
		       root, comm);
    }
    
    /**
     * @brief MPI_Scatterv on std::collections. Send only portions of buff to ranks
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
	    return variables_per_rank[getRank(comm)];
	  }()) == recvbuff.size());
      
      return MPI_Scatterv(sendbuf.data(), variables_per_rank.data(), offsets.data(),
			  detail::mpi_t<typename T::value_type>::type,
			  recvbuff.data(), recvbuff.size(),
			  detail::mpi_t<typename T::value_type>::type, root, comm);
    }



    /**
     *  Inverts the mapping to map[global_index] = {local_indices...}
     *
     *  multiple local_indices may point to the same global index
     *
     *  vector_map = [1 4 3 2]
     *  inverse_map = {{ 1, 0}, {4, 1}, {3, 2}, {2,4}}
     *
     *
     */
    template <typename T>
    auto inverseMap (T & vector_map) {
      std::unordered_map<typename T::value_type, T> map;
      typename T::size_type counter = 0;
      for (auto v : vector_map) {
	if (map.find(v) != map.end()) {
	  map[v].push_back(counter);
	} else {
	  // initialize map[v]
	  map[v] = T {counter};
	}
	counter++;
      }
      // sort the map
      for (auto & [k, v] : map) {
	std::sort(v.begin(), v.end());
      }
      
      return map;
    }
    
    /**
     * @brief given a map of local_ids and global_ids determine send and recv communications
     *
     * @param[in] local_ids maps global_ids to local_ids for this rank. Note the values need to be sorted
     * @param[in] all_global_local_ids This is the global vector of global ids of each rank concatenated
     * @param[in] offsets These are the inclusive offsets of the concatenated vector designated by the number of ids per rank
     * @return An unordered map of recv[rank] = {our_rank's local ids}, send[rank] = {our local id to send}
     *
     */
    
    template <typename T, typename M, typename I>
    std::tuple<std::unordered_map<int, T>, std::unordered_map<int, T>>
    generateSendRecievePerRank(M local_ids,
			       T & all_global_local_ids,
			       I & offsets, MPI_Comm comm = MPI_COMM_WORLD) {
      int my_rank = getRank(comm);
      // Go through global_local_ids looking for local_ids and add either to send_map or recv_map
      typename I::value_type current_rank = 0;

      std::unordered_map<int, T> recv;
      std::unordered_map<int, T> send;
      
      for (const auto & global_local_id : all_global_local_ids) {
	// keep current_rank up-to-date
	auto global_offset = &global_local_id - &all_global_local_ids.front();
	if (static_cast<typename I::value_type>(global_offset) == offsets[current_rank+1]) {
	  current_rank++;
	}

	typename M::iterator found;
	// skip if it's our rank
	if (current_rank != my_rank &&
	    ((found = local_ids.find(global_local_id)) != local_ids.end())) {
	  // The global_local_id is one of ours check to see if we need to send

	  if (current_rank < my_rank) {
	    
	    // append local_id to variables to send to this rank
	    send[current_rank].insert(send[current_rank].end(),
				      (found->second).begin(),
				      (found->second).end());
	    
	    // erase it from our local_ids copy since we've found where to send it
	    local_ids.erase(found);
	  } else if (current_rank > my_rank) {
	    // check to see if we already will recieve data from this rank
	    // we are already recieving data from this rank
	    recv[current_rank].push_back(found->second[0]);

	  }
	  
	}

	// check if local_ids.size() == 0, this case can only occur if all of the local_ids are owned by another MPI_task
	if (local_ids.size() == 0) {
	  break;
	}
      }

      return std::make_tuple(recv, send);
    }

    template <typename T>
    int Irecv(T & buf, int send_rank, MPI_Request * request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      std::cout << "Irecv " << getRank(comm) << ":" << buf.size() << " " << send_rank << " " << tag << std::endl;
      return MPI_Irecv(buf.data(),
		       buf.size(),
		       detail::mpi_t<typename T::value_type>::type,		       
		       send_rank, tag,
		       comm, request);
    }

    template <typename T>
    int Isend(T & buf, int recv_rank, MPI_Request * request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      std::cout << "Isend " << getRank(comm) << ":" << buf.size() << " " << recv_rank << " " << tag << std::endl;
      
      return MPI_Isend(buf.data(), buf.size(),
		       detail::mpi_t<typename T::value_type>::type,		       
		       recv_rank, tag,
		       comm, request);
    }

    int Waitall(std::vector<MPI_Request> & requests, std::vector<MPI_Status> & status) {
      return MPI_Waitall(requests.size(), requests.data(), status.data());
    }

    
    /**
     * @brief transfer data to owners
     */
    template <typename V, typename T>
    auto sendToOwners(std::unordered_map<int, T> & recv, std::unordered_map<int, T> & send,
		      const V & local_data, MPI_Comm comm = MPI_COMM_WORLD) {
      // initiate Irecv first requests
      std::vector<MPI_Request> requests;
      std::unordered_map<int, V> recv_data;
      for (auto [recv_rank, recv_rank_vars]  : recv) {
	// allocate space to recieve
	recv_data[recv_rank] = V(recv_rank_vars.size());
	requests.push_back(MPI_Request());
	// initiate recvieve from rank
	Irecv(recv_data[recv_rank], recv_rank, &requests.back());
      }

      MPI_Barrier(comm);
      std::unordered_map<int, V> send_data;
      for (auto [send_to_rank, send_rank_vars]  : send) {
	send_data[send_to_rank] = V();
	for (auto s : send_rank_vars) {
	  send_data[send_to_rank].push_back(local_data[s]);
	}
	requests.push_back(MPI_Request());
	// initiate recvieve from rank
	Isend(send_data[send_to_rank], send_to_rank, &requests.back());
      }

      std::vector<MPI_Status> stats(requests.size());
      auto error = Waitall(requests, stats);
      if (error != MPI_SUCCESS)
	std::cout << "sendToOwner issue : " << error << std::endl;

      return recv_data;      
    }

    /**
     * @brief transfer back data in reverse from sendToOwners
     */
    template <typename V, typename T>
    auto returnToSender(std::unordered_map<int, T> & recv, std::unordered_map<int, T> & send,
		      const V & local_data, MPI_Comm comm = MPI_COMM_WORLD) {
      // initiate Irecv first requests
      std::vector<MPI_Request> requests;

      
      std::unordered_map<int, V> send_data;
      for (auto [send_to_rank, send_rank_vars]  : send) {
	// populate data to send
	send_data[send_to_rank] = V(send_rank_vars.size());
	requests.push_back(MPI_Request());
	// initiate recvieve from rank
	Irecv(send_data[send_to_rank], send_to_rank, &requests.back());
      }

      
      MPI_Barrier(comm);
      std::unordered_map<int, V> recv_data;
      for (auto [recv_rank, recv_rank_vars]  : recv) {
	// allocate space to recieve
	recv_data[recv_rank] = V();
	for (auto r : recv_rank_vars) {
	  recv_data[recv_rank].push_back(local_data[r]);
	}
	
	requests.push_back(MPI_Request());
	// initiate recvieve from rank
	Isend(recv_data[recv_rank], recv_rank, &requests.back());
      }
      
      std::vector<MPI_Status> stats(requests.size());
      auto error = Waitall(requests, stats);
      if (error != MPI_SUCCESS)
	std::cout << "returnToSender issue : " << error << std::endl;

      return send_data;      
    }


    /**
     * @brief rearrange data so that map[rank]->local_ids and  map[rank] -> V becomes map[local_ids]->values
     *
     * @note doesn't includes local_variable data in the remapped map
     *
     */
    template <typename T, typename V>
    std::unordered_map<typename T::value_type, V>
    remapRecvData(std::unordered_map<int, T> & recv,
		  std::unordered_map<int, V> & recv_data)
    {      
      // recv[from_rank] = {contributions to local indices, will point to first local index corresponding to global index}

      std::unordered_map<typename T::value_type, V> remap;
      for (auto [recv_rank, local_inds] : recv) {
	for (auto &local_ind : local_inds) {
	  auto index = &local_ind - &local_inds.front();
	  auto value = recv_data[recv_rank][index];
	  
	  // local_ind is a key in remap
	  remap[local_ind].push_back( value);
	}
      }
      
      return remap;
    }

    
    /**
     * @brief rearrange data so that map[rank]->local_ids and  map[rank] -> V becomes map[local_ids]->values
     *
     * @note includes local_variable data in the remapped map
     *
     * @return mapping of owned of variable data
     */
    template <typename T, typename V>
    auto remapRecvDataIncludeLocal(std::unordered_map<int, T> & recv,
				   std::unordered_map<int, V> & recv_data,
				   std::unordered_map<typename T::value_type, T> & global_to_local_map,
				   const V & local_variables)
    {      
      auto remap = remapRecvData(recv, recv_data);

      // check to see if recv is empty
      if (recv.size() == 0) {
	// this rank doesn't "own" any variables
	return remap;
      }

      
      // add our own local data to the remapped data
      for (auto [_, local_ids] : global_to_local_map) {
	for (auto local_id : local_ids) {
	  remap[local_ids[0]].push_back(local_variables.at(local_id));
	}
      }

      return remap;
    }

    /**
     * @brief apply reduction operation to recieved data
     *
     */
    template <typename M>
    typename M::mapped_type reduceRecvData(M & remapped_data,
			std::function<typename M::mapped_type::value_type(const typename M::mapped_type &)> reduce_op) {
      typename M::mapped_type reduced_data(remapped_data.size());
      for (auto [local_ind, data_to_reduce] : remapped_data) {
	reduced_data[local_ind] = reduce_op(data_to_reduce);
      }
      return reduced_data;
    }

    /**
     * @brief sum reduction provided for convenience
     */
    template <typename V>
    static typename V::value_type sumOfCollection(const V & collection) {
      typename V::value_type sum = 0;
      for (auto val : collection) {
	sum += val;
      }
      return sum;
    }

    /**
     * @brief selects the first value
     */
    template <typename V>
    static typename V::value_type firstOfCollection(const V & collection) {
      return collection[0];
    }
    
    
    /**
     * @brief remove values in filter that correspond to global_local_ids
     */
    template <typename T>
    auto filterOut(const T & global_local_ids, std::unordered_map<int, std::vector<typename T::size_type>> & filter) {
      std::vector<typename T::size_type> remove_ids;
      for (auto [_, local_ids] : filter) {
	remove_ids.insert(std::end(remove_ids), std::begin(local_ids), std::end(local_ids));
      }
      // sort the ids that we want to remove
      std::sort(remove_ids.begin(), remove_ids.end());

      // filter out all local variables that we are sending
      std::vector<typename T::size_type> local_id_range(global_local_ids.size());
      std::vector<typename T::size_type> filtered(local_id_range.size());
      std::iota(local_id_range.begin(), local_id_range.end(), 0);
      if (remove_ids.size() > 0) {
	auto it = std::set_difference(local_id_range.begin(), local_id_range.end(),
				      remove_ids.begin(), remove_ids.end(),
				      filtered.begin());
	filtered.resize(it - filtered.begin());
      } else {
	filtered = local_id_range;
      }

      // map local ids
      T mapped(filtered.size());
      std::transform(filtered.begin(), filtered.end(),
		     mapped.begin(),
		     [&](auto & v) {
		       return global_local_ids[v];
		     });
      return mapped;      
      
    }
    
  } // namespace utility
  
} // namespace op



#endif
