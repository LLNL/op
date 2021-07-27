#include <mpi.h>
#include "gtest/gtest.h"

#include "op.hpp"
#include "op_config.hpp"

#include "op_debug.hpp"
#include <random>
#include <chrono>

// write variables to disk and read them back in
TEST(test_serializer, WriteAndRead)
{
  int nranks = op::mpi::getNRanks();
  int my_rank = op::mpi::getRank();
  std::vector<std::size_t> rank_variable_labels;
  for (std::size_t i = 0; i < static_cast<std::size_t>(nranks); i++) {
    if (nranks % (my_rank+1) == 0) {
      rank_variable_labels.push_back(i);
    }
  }

  auto comm_pattern = op::AdvancedRegistration(rank_variable_labels);
  op::debug::writeCommPatternToDisk(comm_pattern, my_rank);

  auto comm_pattern_2 = op::debug::readCommPatternFromDisk<decltype(rank_variable_labels)>(my_rank);

  // Deep-compare communication patterns
  for (std::size_t i = 0; i < comm_pattern.owned_variable_list.size(); i++) {
    EXPECT_EQ(comm_pattern.owned_variable_list[i], comm_pattern_2.owned_variable_list[i]);
  }

  for (std::size_t i = 0; i < comm_pattern.local_variable_list.size(); i++) {
    EXPECT_EQ(comm_pattern.local_variable_list[i], comm_pattern_2.local_variable_list[i]);
  }

  // check entries in the recv/send maps
  for (auto [k,_] : comm_pattern.rank_communication.send) {
    for (std::size_t i = 0; i < comm_pattern.rank_communication.send[k].size(); i++) {
      EXPECT_EQ(comm_pattern.rank_communication.send[k][i],
		comm_pattern_2.rank_communication.send[k][i]);
    }
  }

  for (auto [k,_] : comm_pattern.rank_communication.recv) {
    for (std::size_t i = 0; i < comm_pattern.rank_communication.recv[k].size(); i++) {
      EXPECT_EQ(comm_pattern.rank_communication.recv[k][i],
		comm_pattern_2.rank_communication.recv[k][i]);
    }
  }

  // Write some values to disk and read them back
  std::vector<double> local_variables(rank_variable_labels.size());
  for (std::size_t i = 0; i < local_variables.size(); i++) {
    local_variables[i] = rank_variable_labels[i] / (my_rank + 1.1);
  }

  op::debug::writeVectorToDisk(local_variables, my_rank, "local_vector");

  std::vector<double> local_variables_2;
  op::debug::readVectorFromDisk(local_variables_2, my_rank, "local_vector");

  // check if vectors are the same
  for (std::size_t i = 0; i< local_variables.size(); i++) {
    EXPECT_NEAR(local_variables[i], local_variables_2[i], 1.e-3);
  }
}

// Fuzzy test serialization patterns over many different label orderings
TEST(test_serializer, fuzzy_update){

  auto nranks = op::mpi::getNRanks();
  auto my_rank = op::mpi::getRank();

  // This test requires more than one rank
  if (nranks == 1) return;
  
  // we'll go at least 50% overlap
  const int n_local_variables = 4;
  const double overlap = 0.5; // approximate overlap
  const int possible_labels = nranks * n_local_variables;
  const int overlap_max = static_cast<int>(overlap * possible_labels) - 1;
  // chose random integers in the range of 0, overlap_max

  double lower_bound = 0;
  double upper_bound = static_cast<double>(overlap_max);
  std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
  std::default_random_engine re;
  re.seed(std::chrono::system_clock::now().time_since_epoch().count());

  // generate random labels
  std::vector<std::size_t> rank_labels;
  for (int i = 0; i < n_local_variables; i++) {
    std::size_t val = std::lround(unif(re));
    while (std::find(rank_labels.begin(), rank_labels.end(),val) != rank_labels.end()) {
      val = std::lround(unif(re));
    }
    rank_labels.push_back(val);
  }

  op::debug::writeVectorToDisk(rank_labels, my_rank, "rank_labels");
  
  // generate comm_pattern
  auto comm_pattern = op::AdvancedRegistration(rank_labels);

  op::debug::writeCommPatternToDisk(comm_pattern, my_rank);
  
  // time to fuzzy test
  // first we find out how many global variables there are
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
    EXPECT_NEAR(global_values_expect[owned_offsets_[my_rank] + owned_var], owned_data[owned_var], 1.e-8);
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
    auto it = std::find(global_labels.begin(), global_labels.end(), rank_labels[i]);
    auto offset = it - global_labels.begin();
    std::cout << "rank: " << my_rank << " " << offset << ":"  << global_values_mod[offset] << " "
	      << i << ":" << local_variables[i] << std::endl;
    EXPECT_NEAR(global_values_mod[offset], local_variables[i], 1.e-8);
  }
  
  op::debug::writeVectorToDisk(local_data, my_rank, "local_data");
  op::debug::writeVectorToDisk(local_variables, my_rank, "local_variables");
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

