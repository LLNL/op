#include <mpi.h>
#include "gtest/gtest.h"

#include "op.hpp"
#include "op_config.hpp"

#include "op_debug.hpp"

// write variables to disk and read them back in
TEST(test_serializer, writeAndRead)
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


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

