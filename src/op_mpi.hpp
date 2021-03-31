#pragma once
#include <mpi.h>
namespace op {
  
  namespace mpi {
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
    int Allreduce(T * local, T * global, int size, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD) {
      return MPI_Allreduce(local, global, size, mpi::detail::mpi_t<T>::type, op, comm);
    }

        /// MPI_Scatter a vector to all ranks on the communicator
    template <typename T>
    int Broadcast(T & buf, int root = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      return MPI_Bcast(buf.data(),
		       buf.size(),
		       mpi::detail::mpi_t<typename T::value_type>::type,
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
	    return variables_per_rank[mpi::getRank(comm)];
	  }()) == recvbuff.size());
      
      return MPI_Scatterv(sendbuf.data(), variables_per_rank.data(), offsets.data(),
			  mpi::detail::mpi_t<typename T::value_type>::type,
			  recvbuff.data(), recvbuff.size(),
			  mpi::detail::mpi_t<typename T::value_type>::type, root, comm);
    }

    template <typename T>
    int Irecv(T & buf, int send_rank, MPI_Request * request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      std::cout << "Irecv " << mpi::getRank(comm) << ":" << buf.size() << " " << send_rank << " " << tag << std::endl;
      return MPI_Irecv(buf.data(),
		       buf.size(),
		       mpi::detail::mpi_t<typename T::value_type>::type,		       
		       send_rank, tag,
		       comm, request);
    }

    template <typename T>
    int Isend(T & buf, int recv_rank, MPI_Request * request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD) {
      std::cout << "Isend " << mpi::getRank(comm) << ":" << buf.size() << " " << recv_rank << " " << tag << std::endl;
      
      return MPI_Isend(buf.data(), buf.size(),
		       mpi::detail::mpi_t<typename T::value_type>::type,		       
		       recv_rank, tag,
		       comm, request);
    }

    int Waitall(std::vector<MPI_Request> & requests, std::vector<MPI_Status> & status) {
      return MPI_Waitall(requests.size(), requests.data(), status.data());
    }   
    
  } // namespace mpi

} // namespace op
