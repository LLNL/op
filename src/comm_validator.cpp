#include "op.hpp"
#include "op_debug.hpp"

int main(int argc, char * argv[]) {

  MPI_Init(&argc, &argv);

  int my_rank = op::mpi::getRank();
  
  auto comm_pattern = op::debug::readCommPatternFromDisk<std::vector<std::size_t>>(my_rank);

  auto [global_size, variables_per_rank] =
    op::utility::parallel::gatherVariablesPerRank<int>(comm_pattern.owned_variable_list.size());
  std::vector<int> owned_variables_per_rank_ = variables_per_rank;
  auto owned_offsets_ = op::utility::buildInclusiveOffsets(owned_variables_per_rank_);

  // get all the labels globally that are owned (they should be unique)
  auto global_labels = op::utility::parallel::concatGlobalVector(global_size, owned_variables_per_rank_, owned_offsets_, comm_pattern.owned_variable_list);
  
  // build global "truth" values for verification
  std::vector<double> global_values_truth(global_size);
  for (int i = 0; i < global_size; i++) {
    global_values_truth[i] = static_cast<double>(i);
  }

  // on each rank update values by adding shift to global_values that each rank owns
  const double shift = 1.;
  
  auto global_values_expect = global_values_truth;
  for (auto i = owned_offsets_[static_cast<std::size_t>(my_rank)] ; i < owned_offsets_[static_cast<std::size_t>(my_rank) + 1]; i++) {
    global_values_expect[i] += shift;
  }

  // do global serial modification
  auto global_values_mod = global_values_truth;
  for (auto & v : global_values_mod) {
    v += shift;
  }

  std::vector<double> owned_data(comm_pattern.owned_variable_list.size());
  std::vector<double> empty;

  if (my_rank != 0) {
    op::mpi::Scatterv(empty, owned_variables_per_rank_, owned_offsets_, owned_data);
  } else {
    // root node
    op::mpi::Scatterv(global_values_mod, owned_variables_per_rank_, owned_offsets_, owned_data);
  }

  // check values on owned ranks first
  for (int owned_var = 0; owned_var < owned_variables_per_rank_[my_rank]; owned_var++) {
    if (std::abs(global_values_expect[owned_offsets_[my_rank] + owned_var] - owned_data[owned_var]) >= 1.e-8) {
      std::cout << "owned_var mismatch on " << my_rank << " (" << owned_var << ") : " << global_values_expect[owned_offsets_[my_rank] + owned_var] << " " << owned_data[owned_var] << std::endl;
    }
  }

  // check if things are being relayed back to local variables appropriately
  auto global_reduced_map_to_local = op::utility::inverseMap(comm_pattern.local_variable_list);
  
  std::vector<std::size_t> index_map;
  for (auto id : comm_pattern.owned_variable_list) {
    index_map.push_back(global_reduced_map_to_local[id][0]);
  }
  // temporary local_data
  std::vector<double> local_data(comm_pattern.local_variable_list.size());
  std::vector<double> local_variables(comm_pattern.local_variable_list.size());
  op::utility::accessPermuteStore(owned_data, index_map, local_data);

  local_variables = op::ReturnLocalUpdatedVariables(comm_pattern.rank_communication,
						    global_reduced_map_to_local, local_data);

  // check values of variables
  for (std::size_t i = 0; i < local_variables.size(); i++) {
    auto it = std::find(global_labels.begin(), global_labels.end(), comm_pattern.local_variable_list[i]);
    auto offset = it - global_labels.begin();
    if (std::abs(global_values_mod[offset] - local_variables[i]) >= 1.e-8) {
      std::cout << "reduced_update mismatch on " << my_rank << " (" << i << ") : " << global_values_mod[offset] << " " <<local_variables[i] << std::endl;
    }
  }

  

  MPI_Finalize();
  return 0;
}
