
set(test_dependencies mpi op_core)

if (ENABLE_GTEST)

blt_list_append(TO test_dependencies ELEMENTS gtest)

blt_add_executable( NAME TwoCnsts
		    HEADERS ${CMAKE_BINARY_DIR}/include/op/op_config.hpp
		    SOURCES TwoCnsts.cpp
    		    OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
		    DEPENDS_ON nlopt_so ${test_dependencies}
		    FOLDER      op/tests )

foreach(ntask RANGE 1 2)
blt_add_test( NAME "TwoCnsts_${ntask}"
	      COMMAND       TwoCnsts
	      NUM_MPI_TASKS ${ntask})
endforeach()

blt_add_executable( NAME VariableMap
		    HEADERS ${CMAKE_BINARY_DIR}/include/op/op_config.hpp
		    SOURCES VariableMap.cpp
		    OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
		    DEPENDS_ON ${test_dependencies}
		    FOLDER      op/tests )

foreach(ntask RANGE 1 9)
blt_add_test( NAME "VariableMap_${ntask}"
	      COMMAND       VariableMap
	      NUM_MPI_TASKS ${ntask})
endforeach()

blt_add_executable( NAME serializer
		    HEADERS ${CMAKE_BINARY_DIR}/include/op/op_config.hpp
		    SOURCES test_serializer
		    OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
		    DEPENDS_ON ${test_dependencies}
		    FOLDER      op/tests )

foreach(ntask RANGE 1 1)
blt_add_test( NAME "serializer_${ntask}"
	      COMMAND       serializer
	      NUM_MPI_TASKS ${ntask})
endforeach()



endif() # GTEST
