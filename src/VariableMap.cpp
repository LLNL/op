#include "op.hpp"
#include <iostream>

#include "gtest/gtest.h"
#include "op.hpp"
#include "op_config.hpp"
#include "mpi.h"

#include <vector>
#include <algorithm>
#include <sstream>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
  {
    for (auto e : v) {
        os << e << " ";
    }
    return os;
  }

/**
 * This test constructs a simple 1-10 MPI-rank problem
 * 9 variables:
 * 0 1 2 
 * 3 4 5
 * 6 7 8
 *
 * Run through different scenarios
 */


  /**
   * In this case each each partition is assigned a few variables
   *
   * rank i has dv_index % nranks
   *
   * objective: reduce(sum(local_variables))
   * gradient: concat(compute(local_variables))
   * update: happens in parlalel
   */

TEST(VariableMap, density_parallel_update)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int nranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int num_global_vars = 9;

  // this mapping takes us to global decision variables processed on this rank
  std::vector<int> dvs_on_rank;

  // Come up with strided mapping
  for (int i = 0; i < num_global_vars; i++) {
    if (i % nranks == rank) {
      dvs_on_rank.push_back(i);
    }
  }

  std::vector<double> local_variables(dvs_on_rank.begin(), dvs_on_rank.end());
  auto local_lower_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), 0.); };
  auto local_upper_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), static_cast<double>(num_global_vars * 2)); };
  
  op::Vector<std::vector<double>> local_vector(local_variables, local_lower_bounds, local_upper_bounds);
  auto local_obj = [](const std::vector<double> & variables) {
    double sum = 0;
    for (auto v : variables) {
      sum += v;
    }
    return sum;
  };

  auto global_obj = [&](const std::vector<double> & variables) {
    double local_sum = local_obj(variables);
    double global_sum = 0;
    auto error = op::utility::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
    if (error != MPI_SUCCESS) {
      std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
    }
    std::cout << "global sum :" << global_sum << std::endl;
    return global_sum;
  };

  auto local_obj_grad = [](const std::vector<double> & variables) {
    return std::vector<double> (variables.begin(), variables.end());
  };
  
  op::Objective obj (global_obj, local_obj_grad);
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;

  EXPECT_NEAR(obj.Eval(local_vector.data()), 36, 1.e-14); 
  
  // gather global variable information
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector);
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // concat all the variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, local_vector.data());
  
  if (rank == 0) {
    std::cout << "global gradient: "
	      << concatenated_vector << std::endl;
  }

  
  // add the rank variable to all of this rank's variables
  auto update = [&]() {
    std::transform(local_vector.data().begin(), local_vector.data().end(),
		   local_vector.data().begin(),
		   [&](double v) -> double { return v + rank; });
  };

  // Call update
  update();
  
  double local_rank_adj = rank * local_vector.data().size();
  double global_rank_adj = 0.;
  op::utility::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;
 
  EXPECT_NEAR(obj.Eval(local_vector.data()), 36 + global_rank_adj, 1.e-14); 
  
}

/**
 * In this test we perform the global update on rank 0 and scatter new vectors back
 */
TEST(VariableMap, density_serial_update)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int nranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int num_global_vars = 9;

  // this mapping takes us to global decision variables processed on this rank
  std::vector<int> dvs_on_rank;

  // Come up with strided mapping
  for (int i = 0; i < num_global_vars; i++) {
    if (i % nranks == rank) {
      dvs_on_rank.push_back(i);
    }
  }

  std::vector<double> local_variables(dvs_on_rank.begin(), dvs_on_rank.end());
  auto local_lower_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), 0.); };
  auto local_upper_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), static_cast<double>(num_global_vars * 2)); };
  
  op::Vector<std::vector<double>> local_vector(local_variables, local_lower_bounds, local_upper_bounds);
  auto local_obj = [](const std::vector<double> & variables) {
    double sum = 0;
    for (auto v : variables) {
      sum += v;
    }
    return sum;
  };

  auto global_obj = [&](const std::vector<double> & variables) {
    double local_sum = local_obj(variables);
    double global_sum = 0;
    auto error = op::utility::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
    if (error != MPI_SUCCESS) {
      std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
    }
    return global_sum;
  };

  auto local_obj_grad = [](const std::vector<double> & variables) {
    return std::vector<double> (variables.begin(), variables.end());
  };
  
  op::Objective obj (global_obj, local_obj_grad);
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;

  EXPECT_NEAR(obj.Eval(local_vector.data()), 36, 1.e-14); 
  
  // gather global variable information
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector);
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // concat all variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, offsets, local_vector.data());

  // add the rank variable to all of this rank's variables
  auto update = [&]() {
    if (rank == 0) {
      int effective_rank = 0;
      for (typename std::vector<int>::size_type v = 0 ; v < concatenated_vector.size(); v++) {
	if (static_cast<int>(v)==offsets[effective_rank + 1]) {
	  effective_rank++;
	}
	concatenated_vector[v] += effective_rank;
      }
    }
    // Scatter a portion of the results back to their local_vector.data()
    op::utility::Scatterv(concatenated_vector, variables_per_rank, offsets, local_vector.data());
  };

  // Call update and check results
  update();
  
  double local_rank_adj = rank * local_vector.data().size();
  double global_rank_adj = 0.;
  op::utility::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;
 
  EXPECT_NEAR(obj.Eval(local_vector.data()), 36 + global_rank_adj, 1.e-14); 

  
  // Form global local to global id map on rank 0
  auto global_ids_from_global_local_ids =
    op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank);
  
  if (rank == 0 ) {
    std::cout << "global-local ids: " << global_ids_from_global_local_ids << std::endl;
  }  
}

/**
The same configuration as before, except this time we will want to update the variable on the global mapping
 */
TEST(VariableMap, update_serial_global_ids) {
  /*
   * Since partitions are scatted i % nranks == rank
   * if we use our local_to_global mapping we should
   * be able to use thsi relation to get the rank and
   * and perform the update in serial without offsets
   *
   */
  MPI_Barrier(MPI_COMM_WORLD);

  int nranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int num_global_vars = 9;

  // this mapping takes us to global decision variables processed on this rank
  std::vector<int> dvs_on_rank;

  // Come up with strided mapping
  for (int i = 0; i < num_global_vars; i++) {
    if (i % nranks == rank) {
      dvs_on_rank.push_back(i);
    }
  }

  std::vector<double> local_variables(dvs_on_rank.begin(), dvs_on_rank.end());
  auto local_lower_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), 0.); };
  auto local_upper_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), static_cast<double>(num_global_vars * 2)); };
  
  op::Vector<std::vector<double>> local_vector(local_variables, local_lower_bounds, local_upper_bounds);
  auto local_obj = [](const std::vector<double> & variables) {
    double sum = 0;
    for (auto v : variables) {
      sum += v;
    }
    return sum;
  };

  auto global_obj = [&](const std::vector<double> & variables) {
    double local_sum = local_obj(variables);
    double global_sum = 0;
    auto error = op::utility::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
    if (error != MPI_SUCCESS) {
      std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
    }
    return global_sum;
  };

  auto local_obj_grad = [](const std::vector<double> & variables) {
    return std::vector<double> (variables.begin(), variables.end());
  };
  
  op::Objective obj (global_obj, local_obj_grad);
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;

  EXPECT_NEAR(obj.Eval(local_vector.data()), 36, 1.e-14); 
  
  // gather global variable information
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector);
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // Form global local to global id map on rank 0
  auto global_ids_from_global_local_ids =
    op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank);
  
  if (rank == 0 ) {
    std::cout << "global-local ids: " << global_ids_from_global_local_ids << std::endl;
  }  
  
  // concat all variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, offsets, local_vector.data());
  
  if (rank == 0) {
    concatenated_vector = op::utility::selectIndexMap(concatenated_vector, global_ids_from_global_local_ids);
  }
		   
  // add the rank variable to all of this rank's variables
  auto update = [&]() {
    if (rank == 0) {
      for (typename std::vector<int>::size_type v = 0 ; v < concatenated_vector.size(); v++) {
	concatenated_vector[v] += v % nranks;
      }
    }
    concatenated_vector = op::utility::selectIndexMap(concatenated_vector, global_ids_from_global_local_ids);
    // Scatter a portion of the results back to their local_vector.data()
    op::utility::Scatterv(concatenated_vector, variables_per_rank, offsets, local_vector.data());
  };

  // Call update and check results
  update();
  
  double local_rank_adj = rank * local_vector.data().size();
  double global_rank_adj = 0.;
  op::utility::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;
 
  EXPECT_NEAR(obj.Eval(local_vector.data()), 36 + global_rank_adj, 1.e-14); 
  
}

int main(int argc, char*argv[])
{

  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  
  auto results = RUN_ALL_TESTS();

  MPI_Finalize();
  return results;
}
