// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (c) 2021-2022, Lawrence Livermore National Security, LLC
// All rights reserved.

#pragma once

#include "op.hpp"
#include "op_mpi.hpp"
#include "op_utility.hpp"
#include <fstream>
#include <sstream>

namespace op {
  /// This namespace includes several methods for debugging parallel communication patterns
  namespace debug {

    /**
     * @brief Write vector to disk. Files are named vector_string.rank
     *
     * @param[in] vector rank-local optimization variables
     * @param[in] local_rank current_rank
     * @param[in] local_vector_string prefix for file
     */
    template <typename VectorType>
    void writeVectorToDisk(const std::vector<VectorType> & vector, int local_rank, std::string local_vector_string)
    {     
      std::stringstream local_vector_file;
      local_vector_file << local_vector_string << "." << local_rank;
      std::ofstream local_vector(local_vector_file.str());
      for (auto v : vector) {
	local_vector << v << std::endl;
      }
      local_vector.close();           
    }

    
    /**
     * @brief Write local variables to disk. Files are named local_variable_string.rank
     *
     * @param[in] op_variables rank-local optimization variables
     * @param[in] local_rank current_rank
     * @param[in] local_variable_string prefix for file
     */
    template <typename VectorType>
    void writeVariablesToDisk(const op::Vector<VectorType> & op_variables, int local_rank, std::string local_variable_string = "local_variables")
    {
      writeVectorToDisk(op_variables.data(), local_rank, local_variable_string);
    }

    /**
     * @brief Read vector to disk. Files are named vector_string.rank
     *
     * @param[in] vector rank-local optimization variables
     * @param[in] local_rank current_rank
     * @param[in] local_vector_string prefix for file
     */
    template <typename VectorType>
    void readVectorFromDisk(std::vector<VectorType> & vector, int local_rank, std::string local_vector_string)
    {     
      std::stringstream local_vector_file;
      local_vector_file << local_vector_string << "." << local_rank;
      std::ifstream local_vector(local_vector_file.str());
      while (local_vector.good() && !local_vector.eof()) {
	VectorType v;
	local_vector >> v;
	vector.push_back(v);
      }
      local_vector.close();           
    }
    
    /**
     * @brief Write comm_pattern to disk. Files are named comm_pattern_string.rank
     *
     * @param[in] comm_pattern rank-local comm pattern
     * @param[in] local_rank current_rank
     * @param[in] comm_pattern_string prefix for file
     */

    template <typename VectorIndexType>
    void writeCommPatternToDisk(op::utility::CommPattern<VectorIndexType> & comm_pattern, int local_rank, std::string comm_pattern_string = "pattern")
    {
      std::stringstream comm_pattern_file_name;
      comm_pattern_file_name << comm_pattern_string << "." << local_rank;
      std::ofstream comm_pattern_file(comm_pattern_file_name.str());
    
      auto rank_comm = comm_pattern.rank_communication;

      comm_pattern_file << "rank communication:" << std::endl;
      comm_pattern_file << "send:" << std::endl;
      for (auto [rank, values] : rank_comm.send) {
	comm_pattern_file << rank << std::endl << " : " << std::endl;
	for (auto v : values) {
	  comm_pattern_file << v << std::endl;
	}
	comm_pattern_file << std::endl;
      }

      comm_pattern_file << "recv:" << std::endl;
      for (auto [rank, values] : rank_comm.recv) {
	comm_pattern_file << rank << std::endl << " : " << std::endl;
	for (auto v : values) {
	  comm_pattern_file << v << std::endl;
	}
	comm_pattern_file << std::endl;
      }

      comm_pattern_file << "owned_variable_list:" << std::endl;
      for (auto v : comm_pattern.owned_variable_list) {
	comm_pattern_file << v << std::endl;
      }
      comm_pattern_file << std::endl;

      comm_pattern_file << "local_variable_list:" << std::endl;
      for (auto v : comm_pattern.local_variable_list) {
	comm_pattern_file << v << std::endl;
      }
      comm_pattern_file << std::endl;
      comm_pattern_file.close();      
    }

    /**
     * @brief Read comm_pattern to disk. Files are named comm_pattern_string.rank
     *
     * @param[in] comm_pattern rank-local comm pattern
     * @param[in] local_rank current_rank
     * @param[in] comm_pattern_string prefix for file
     */

    template <typename VectorIndexType>
    op::utility::CommPattern<VectorIndexType> readCommPatternFromDisk(int local_rank, std::string comm_pattern_string = "pattern")
    {
      op::utility::CommPattern<VectorIndexType> comm_pattern;
      
      std::stringstream comm_pattern_file_name;
      comm_pattern_file_name << comm_pattern_string << "." << local_rank;
      std::ifstream comm_pattern_file(comm_pattern_file_name.str());
    
      auto rank_comm = comm_pattern.rank_communication;

      std::string buffer;

      // rank communication:
      std::getline(comm_pattern_file, buffer);
      buffer.clear();
      // Check if send map exists
      std::getline(comm_pattern_file, buffer);
      
      if (buffer == "send:") {
	// Read in send map
	while(comm_pattern_file.good()) {
	  buffer.clear();
	  std::getline(comm_pattern_file, buffer);
	  if (buffer == "recv:") break;
	  int rank = std::stoi(buffer);
	  // skip " : "
	  std::getline(comm_pattern_file, buffer);
	  buffer.clear();
	  std::getline(comm_pattern_file, buffer);  
	  // read until we reach \n \n
	  VectorIndexType indices;
	  while (comm_pattern_file.good() && buffer.length() > 0) {
	    std::size_t index = std::stoul(buffer);
	    indices.push_back(index);
	    buffer.clear();
	    std::getline(comm_pattern_file, buffer);
	  }
	  // add to map
	  comm_pattern.rank_communication.send[rank] = indices;
	}
      }

      if (buffer == "recv:") {
	// Read in recv map
	while(comm_pattern_file.good()) {
	  buffer.clear();
	  std::getline(comm_pattern_file, buffer);
	  if (buffer == "owned_variable_list:") break;
	  int rank = std::stoi(buffer);
	  // skip " : "
	  std::getline(comm_pattern_file, buffer);
	  buffer.clear();
	  std::getline(comm_pattern_file, buffer);
	  // read until we reach \n \n
	  VectorIndexType indices;
	  while (comm_pattern_file.good() && buffer.length() > 0) {
	    std::size_t index = std::stoul(buffer);
	    indices.push_back(index);
	    buffer.clear();
	    std::getline(comm_pattern_file, buffer);	    
	  }
	  // add to map
	  comm_pattern.rank_communication.recv[rank] = indices;
	}
      }

      // Read in owned_variable_list
      buffer.clear();
      std::getline(comm_pattern_file, buffer);	
      while(comm_pattern_file.good() && buffer.length() > 0) {
	if constexpr (std::is_same_v<typename VectorIndexType::value_type, int>) {
	    comm_pattern.owned_variable_list.push_back(std::stoi(buffer));
	  } else if constexpr(std::is_same_v<typename VectorIndexType::value_type, std::size_t>){
	    comm_pattern.owned_variable_list.push_back(std::stoul(buffer));
	  }
	buffer.clear();
	std::getline(comm_pattern_file, buffer);	
      }

      // Read in local_variable_list
      buffer.clear();
      std::getline(comm_pattern_file, buffer); // skip local_variable_list: string
      buffer.clear();
      std::getline(comm_pattern_file, buffer); 
      while(comm_pattern_file.good() && buffer.length() > 0) {
	if constexpr (std::is_same_v<typename VectorIndexType::value_type, int>) {
	    comm_pattern.local_variable_list.push_back(std::stoi(buffer));
	  } else if constexpr(std::is_same_v<typename VectorIndexType::value_type, std::size_t>){
	    comm_pattern.local_variable_list.push_back(std::stoul(buffer));
	  }
	buffer.clear();
	std::getline(comm_pattern_file, buffer);	
      }
      
      comm_pattern_file.close();
      return comm_pattern;
    }
    
  } // debug namespace
} // op namespace
