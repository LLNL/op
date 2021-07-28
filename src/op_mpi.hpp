#pragma once
#include <mpi.h>
namespace op {

/// template MPI namespace
namespace mpi {
/// MPI related type traits
namespace detail {

// default template
template <typename T>
struct mpi_t {
  static const MPI_Datatype type = MPI_BYTE;
};

template <>
struct mpi_t<double> {
  static const MPI_Datatype type = MPI_DOUBLE;
};

template <>
struct mpi_t<int> {
  static const MPI_Datatype type = MPI_INT;
};

template <>
struct mpi_t<unsigned long> {
  static const MPI_Datatype type = MPI_UNSIGNED_LONG;
};

template <typename T, typename SFINAE = void>
struct has_data : std::false_type {
};

template <typename T>
struct has_data<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {
};

template <typename T, typename SFINAE = void>
struct has_size : std::false_type {
};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().data())>> : std::true_type {
};

}  // namespace detail

/// Get rank
int getRank(MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

/// Get number of ranks
int getNRanks(MPI_Comm comm = MPI_COMM_WORLD)
{
  int nranks;
  MPI_Comm_size(comm, &nranks);
  return nranks;
}

/**
 * @brief All reduce a single element across all ranks in a communicator
 *
 * @param[in] local element contribution to reduce
 * @param[out] global element to reduce to
 * @param[in] operation MPI_Op
 * @param[in] comm MPI communicator
 */

template <typename T>
std::enable_if_t<!(detail::has_data<T>::value && detail::has_size<T>::value), int> Allreduce(
    T& local, T& global, MPI_Op operation, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Allreduce(&local, &global, 1, mpi::detail::mpi_t<T>::type, operation, comm);
}

/**
 * @brief All reduce std::collections across all ranks in a communicator
 *
 * @param[in] local std::collection contribution to reduce
 * @param[out] global std::collection to reduce to
 * @param[in] operation MPI_Op
 * @param[in] comm MPI communicator
 */

template <typename T>
std::enable_if_t<(detail::has_data<T>::value && detail::has_size<T>::value), int> Allreduce(
    T& local, T& global, MPI_Op operation, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Allreduce(local.data(), global.data(), local.size(), mpi::detail::mpi_t<typename T::value_type>::type,
                       operation, comm);
}

/**
 * @brief Broadcast a single element to all ranks on the communicator
 *
 * @param[in] buf std::collection to broadcast
 * @param[in] root Root rank
 * @param[in] comm MPI communicator
 */
template <typename T>
std::enable_if_t<!(detail::has_data<T>::value && detail::has_size<T>::value), int> Broadcast(
    T& buf, int root = 0, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Bcast(&buf, 1, mpi::detail::mpi_t<T>::type, root, comm);
}

/**
 * @brief Broadcast a vector to all ranks on the communicator
 *
 * @param[in] buf std::collection to broadcast
 * @param[in] root Root rank
 * @param[in] comm MPI communicator
 */
template <typename T>
std::enable_if_t<(detail::has_data<T>::value && detail::has_size<T>::value), int> Broadcast(
    T& buf, int root = 0, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Bcast(buf.data(), static_cast<int>(buf.size()), mpi::detail::mpi_t<typename T::value_type>::type, root,
                   comm);
}

/**
 * @brief gathers a local collections from all ranks on all ranks on a communicator
 *
 * @param[in] buf rank-local std::collection to gather
 * @param[out] values_on_rank the globally-colelcted std::collection
 * @param[in] size_on_rank Number of variables per rank
 * @param[in] offsets_on_rank Offsets in values_on_rank corresponding to a given rank
 * @param[in] comm MPI Communicator
 */
template <typename T>
int Allgatherv(T& buf, T& values_on_rank, std::vector<int>& size_on_rank, std::vector<int>& offsets_on_rank,
               MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Allgatherv(buf.data(), static_cast<int>(buf.size()), detail::mpi_t<typename T::value_type>::type,
                        values_on_rank.data(), size_on_rank.data(), offsets_on_rank.data(),
                        detail::mpi_t<typename T::value_type>::type, comm);
}

/**
 * @brief gathers a local collections from all ranks only on the root rank
 *
 * @param[in] buf rank-local std::collection to gather
 * @param[out] values_on_rank the globally-colelcted std::collection
 * @param[in] size_on_rank Number of variables per rank
 * @param[in] offsets_on_rank Offsets in values_on_rank corresponding to a given rank
 * @param[in] root root rank
 * @param[in] comm MPI Communicator
 */

template <typename T>
int Gatherv(T& buf, T& values_on_rank, std::vector<int>& size_on_rank, std::vector<int>& offsets_on_rank, int root = 0,
            MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Gatherv(buf.data(), static_cast<int>(buf.size()), detail::mpi_t<typename T::value_type>::type,
                     values_on_rank.data(), size_on_rank.data(), offsets_on_rank.data(),
                     detail::mpi_t<typename T::value_type>::type, root, comm);
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
int Scatterv(T& sendbuf, std::vector<int>& variables_per_rank, std::vector<int>& offsets, T& recvbuff, int root = 0,
             MPI_Comm comm = MPI_COMM_WORLD)
{
  // only check the size of the recv buff in debug mode
  assert(static_cast<typename T::size_type>(
             [&]() { return variables_per_rank[static_cast<std::size_t>(mpi::getRank(comm))]; }()) == recvbuff.size());

  return MPI_Scatterv(sendbuf.data(), variables_per_rank.data(), offsets.data(),
                      mpi::detail::mpi_t<typename T::value_type>::type, recvbuff.data(),
                      static_cast<int>(recvbuff.size()), mpi::detail::mpi_t<typename T::value_type>::type, root, comm);
}

/**
 * @brief Recieve a buffer from a specified rank and create a handle for the MPI_Request
 *
 * @param[out] buf std::collection to recieve into
 * @param[in] send_rank The rank sending the information
 * @param[out] request the MPI request handle
 * @param[in] tag A tag to identify the communication message
 * @param[in] comm MPI communicator
 */
template <typename T>
int Irecv(T& buf, int send_rank, MPI_Request* request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Irecv(buf.data(), static_cast<int>(buf.size()), mpi::detail::mpi_t<typename T::value_type>::type,
                   send_rank, tag, comm, request);
}

/**
 * @brief Send a buffer to a specified rank and create a handle for the MPI_Request
 *
 * @param[in] buf std::collection to send
 * @param[in] recv_rank The rank recieving the data
 * @param[out] request the MPI request handle
 * @param[in] tag A tag to identify the communication message
 * @param[in] comm MPI communicator
 */

template <typename T>
int Isend(T& buf, int recv_rank, MPI_Request* request, int tag = 0, MPI_Comm comm = MPI_COMM_WORLD)
{
  return MPI_Isend(buf.data(), static_cast<int>(buf.size()), mpi::detail::mpi_t<typename T::value_type>::type,
                   recv_rank, tag, comm, request);
}

/**
 * @brief A wrapper to MPI_Waitall to wait for all the requests to be fulfilled
 *
 * @param[in] requests A vector of MPI_Request handles
 * @param[in] status A vector MPI_Status for each of the handles
 */

int Waitall(std::vector<MPI_Request>& requests, std::vector<MPI_Status>& status)
{
  return MPI_Waitall(static_cast<int>(requests.size()), requests.data(), status.data());
}

int CreateAndSetErrorHandler(MPI_Errhandler& newerr, void (*err)(MPI_Comm* comm, int* err, ...),
                             MPI_Comm        comm = MPI_COMM_WORLD)
{
  MPI_Comm_create_errhandler(err, &newerr);
  return MPI_Comm_set_errhandler(comm, newerr);
}

}  // namespace mpi

}  // namespace op
