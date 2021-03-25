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

TEST(VariableMap, density)
{
  MPI_Barrier(MPI_COMM_WORLD);
  /**
   * In this case each each partition is assigned a few variables
   *
   * rank i has dv_index % nranks
   *
   * objective: reduce(sum(local_variables))
   * gradient: concat(compute(local_variables))
   * update: global
   */

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

  // gather global variable information
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector);
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // concat all the gradients
  auto concatenated_vector = op::utility::concatGlobalVector(global_size, variables_per_rank, local_vector);
  if (rank == 0) {
    std::cout << "global gradient: "
	      << concatenated_vector << std::endl;
  }

  
  // add the rank variable to all of this ranks' variables
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

int main(int argc, char*argv[])
{

  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  
  auto results = RUN_ALL_TESTS();

  MPI_Finalize();
  return results;
}
