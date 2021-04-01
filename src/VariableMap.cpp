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
void ARRAY_EXPECT_EQ(const T & check, const T & values) {
  EXPECT_EQ(check.size(), values.size());
  for (int i = 0; i < check.size(); i++) {
    EXPECT_EQ(check[i], values[i]);
  }
}

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
    auto error = op::mpi::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
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
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector.data().size());
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // concat all the variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, local_vector.data(), false);
  
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
  op::mpi::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
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
    auto error = op::mpi::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
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
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector.data().size());
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // concat all variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, offsets, local_vector.data(), false);

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
    op::mpi::Scatterv(concatenated_vector, variables_per_rank, offsets, local_vector.data());
  };

  // Call update and check results
  update();
  
  double local_rank_adj = rank * local_vector.data().size();
  double global_rank_adj = 0.;
  op::mpi::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
  std::cout << "rank " << rank << " : " << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;
 
  EXPECT_NEAR(obj.Eval(local_vector.data()), 36 + global_rank_adj, 1.e-14); 

  
  // Form global local to global id map on rank 0
  auto global_ids_from_global_local_ids =
    op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank, false);
  
  if (rank == 0 ) {
    std::cout << "global-local ids: " << global_ids_from_global_local_ids << std::endl;
    std::vector<int> global_id_check;
    for (int r = 0; r < nranks; r++) {
      for (int i = 0; i < num_global_vars; i++) {
	if (i % nranks == r) {
	  global_id_check.push_back(i);
	}
      }
    }
    
    ARRAY_EXPECT_EQ(global_ids_from_global_local_ids, global_id_check);
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
    auto error = op::mpi::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
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
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(local_vector.data().size());
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // Form global local to global id map on rank 0
  auto global_ids_from_global_local_ids =
    op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank, false);
  
  if (rank == 0 ) {
    std::cout << "global-local ids: " << global_ids_from_global_local_ids << std::endl;
  }  
  
  // concat all variables
  auto concatenated_vector =
    op::utility::concatGlobalVector(global_size, variables_per_rank, offsets, local_vector.data(), false);
  
  if (rank == 0) {
    concatenated_vector = op::utility::accessPermuteStore(concatenated_vector, global_ids_from_global_local_ids, -1);
  }
		   
  // add the rank variable to all of this rank's variables
  auto update = [&]() {
    if (rank == 0) {
      for (typename std::vector<int>::size_type v = 0 ; v < concatenated_vector.size(); v++) {
	concatenated_vector[v] += v % nranks;
      }
      std::cout << "concatenated_vector: " << concatenated_vector << std::endl;

      concatenated_vector = op::utility::permuteAccessStore(concatenated_vector, global_ids_from_global_local_ids);
      std::cout << "concatenated_vector re-indexed: " << concatenated_vector << std::endl;
    }
    // Scatter a portion of the results back to their local_vector.data()
    op::mpi::Scatterv(concatenated_vector, variables_per_rank, offsets, local_vector.data());
  };

  // Call update and check results
  update();
  
  double local_rank_adj = rank * local_vector.data().size();
  double global_rank_adj = 0.;
  op::mpi::Allreduce(&local_rank_adj, &global_rank_adj, 1, MPI_SUM);
  
  std::cout << "rank " << rank << " : " << local_vector.data() << ":" << obj.Eval(local_vector.data())
	    << ": " << obj.EvalGradient(local_vector.data()) << std::endl;
 
  EXPECT_NEAR(obj.Eval(local_vector.data()), 36 + global_rank_adj, 1.e-14); 
  
}

/* 
   This test case has the same partioning locally. 
   However now every former global id , i, corresponds to reduce_i = floor(i/3)

   Instead of 9 variables now we have 3
   * 3 variables:
   * 1 1 1 
   * 2 2 2
   * 3 3 3

   */
TEST(VariableMap, update_serial_reduced_variables) {
  MPI_Barrier(MPI_COMM_WORLD);

  int nranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr int num_global_ids = 9;
  constexpr int num_global_vars = 3;

  // this mapping takes us to global decision variables processed on this rank
  std::vector<typename std::vector<int>::size_type> dvs_on_rank;

  // Come up with strided mapping. Only add global variables once
  for (int i = 0; i < num_global_ids; i++) {
    if (i % nranks == rank) {
      auto global_var_id =static_cast<typename decltype(dvs_on_rank)::size_type>(i / num_global_vars);
      if (dvs_on_rank.size() == 0 || dvs_on_rank.back() != global_var_id) {
	dvs_on_rank.push_back(global_var_id);
      }
    }
  }
 
  // gather global variable information
  auto [global_size, variables_per_rank] = op::utility::gatherVariablesPerRank<int>(dvs_on_rank.size());
  std::cout << "number of global variables:" << global_size << ": "
	    << variables_per_rank << std::endl;

  auto offsets = op::utility::buildInclusiveOffsets(variables_per_rank);
  std::cout << "offsets :" << offsets << std::endl;

  // Form global local to global id map on rank 0
  // auto global_ids_from_global_local_ids =
  //   op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank, false);

  // // Scatter it back to everyone. A global variable may be dependent on more than one rank
  // global_ids_from_global_local_ids.resize(global_size);
  // auto error = op::mpi::Broadcast(global_ids_from_global_local_ids);
  // if (error != MPI_SUCCESS)
  //   std::cout << error << std::endl;
  
  // Form ids and give to everyone
  auto global_ids_from_global_local_ids =
    op::utility::concatGlobalVector(global_size, variables_per_rank, dvs_on_rank);
  
  std::cout << "rank global_ids " << rank << " : " << global_ids_from_global_local_ids << std::endl;
  
  //  determine owner dependencies -> create dependency graph
  //  first rank with a global_id is the owner
  auto global_ids_to_local = op::utility::inverseMap(dvs_on_rank);
  auto recv_send_info =
    op::utility::generateSendRecievePerRank(global_ids_to_local,
  					    global_ids_from_global_local_ids, offsets);

  for (auto &r : recv_send_info.recv) {
    std::cout << "recv " << rank << " : " << r.first << " : " << r.second << std::endl;
  }

  for (auto &s : recv_send_info.send) {
    std::cout << "send " << rank << " : " << s.first << " : " << s.second << std::endl;
  }

  // filter out entries that correspond to send to get our local variables that we own
  auto reduced_dvs_on_rank = op::utility::filterOut(dvs_on_rank, recv_send_info.send);

  std::cout << "reduced dvs on rank " << rank << " : " << reduced_dvs_on_rank << std::endl;
  // get all of this information on all the ranks
  auto [reduced_global_size, reduced_variables_per_rank] = op::utility::gatherVariablesPerRank<int>(reduced_dvs_on_rank.size());
  EXPECT_EQ(reduced_global_size, num_global_vars);

  // The local variables are dvs_on_rank.
  std::vector<double> local_variables(dvs_on_rank.begin(), dvs_on_rank.end());

  // However the actual variables we send to rank 0 are reduced_dvs_on_rank

  auto local_lower_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), 0.); };
  auto local_upper_bounds = [&]() { return std::vector<double> (dvs_on_rank.size(), static_cast<double>(num_global_vars * 2)); };
  
  op::Vector<std::vector<double>> reduced_local_vector(local_variables, local_lower_bounds, local_upper_bounds);

  // When calculating the objective, every rank calculates it's local objective
  auto local_obj = [](const std::vector<double> & local_variables) {
    double sum = 0;
    for (auto v : local_variables) {
      sum += v/num_global_ids;
    }
    return sum;
  };

  auto global_obj = [&](const std::vector<double> & local_variables) {
    double local_sum = local_obj(local_variables);
    double global_sum = 0;
    auto error = op::mpi::Allreduce(&local_sum, &global_sum, 1, MPI_SUM);
    if (error != MPI_SUCCESS) {
      std::cout << "MPI_Error" << __FILE__ << __LINE__ << std::endl;
    }
    std::cout << "global sum :" << global_sum << std::endl;
    return global_sum;
  };

  // For the gradients things get more interesting
  // First compute the local_obj_gradient from this rank
  auto local_obj_grad = [](const std::vector<double> & local_variables) {
    std::vector<double> grad(local_variables.size());
    std::transform(local_variables.begin(), local_variables.end(), grad.begin(),
		   [](double v ) {
		     return v/num_global_ids;
		   });
    return grad;
  };
  // We want the reduced_local_obj_grad
  auto reduced_local_obj_grad = [&] (const std::vector<double> & local_variables) {
    // First we send any local gradient information to the ranks that "own" the variables
    auto local_obj_gradient = local_obj_grad(local_variables);
    auto recv_data = op::utility::sendToOwners(recv_send_info, local_obj_gradient);
    auto combine_data = op::utility::remapRecvDataIncludeLocal(recv_send_info.recv, recv_data, global_ids_to_local, local_obj_gradient);
    std::vector<double> reduced_local_variables =
    op::utility::reduceRecvData(combine_data,
				op::utility::reductions::sumOfCollection<typename decltype(combine_data)::mapped_type>);
    return reduced_local_variables;
  };

  auto reduced_local_grad = reduced_local_obj_grad(local_variables);

  std::cout << "reduced_local_grad " << rank << " : " << reduced_local_grad << std::endl;

  // Test back propagation of updated variables
  std::vector<double> reduced_updated_values(dvs_on_rank.size(), rank);
  // currently this output is std::vector[rank] = local index values
  // for the variables a rank owns.. update() should propagate those
  // for the variables an update does not own.. they will be in returned_data
  // returned_remapped_data is a map[local_ids] -> values
  // we want to write it back into our local variable array
  
  auto returned_data = op::utility::returnToSender(recv_send_info, reduced_updated_values);
  auto returned_remapped_data = op::utility::remapRecvDataIncludeLocal(recv_send_info.send, returned_data, global_ids_to_local, reduced_updated_values);
  std::vector<double> updated_local_variables;
  if (recv_send_info.send.size() == 0) {
    // we own all the variables
    updated_local_variables = reduced_updated_values;
  } else {
    updated_local_variables =
      op::utility::reduceRecvData(returned_remapped_data,
				  op::utility::reductions::firstOfCollection<typename decltype(returned_remapped_data)::mapped_type>);
  }

  std::cout << "reduced variables " << rank << " : " << reduced_updated_values << std::endl;
  std::cout << "updated variables " << rank << " : " << updated_local_variables << std::endl;

  // go through local variables and check their values ..which should be who owns them
  for (auto [send_to_rank, send_vars] : recv_send_info.send) {
    for (auto send_var : send_vars) {
      EXPECT_EQ(send_to_rank, updated_local_variables[send_var]);
    }
  }

  for (auto owned_var : reduced_dvs_on_rank) {
    auto index = std::find(dvs_on_rank.begin(), dvs_on_rank.end(), owned_var) - dvs_on_rank.begin();
    std::cout << "checking owned " << rank << " : " << owned_var << " at " << index << std::endl;
    EXPECT_EQ(rank, updated_local_variables[index]);
  }

  
}

int main(int argc, char*argv[])
{

  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  
  auto results = RUN_ALL_TESTS();

  MPI_Finalize();
  return results;
}
