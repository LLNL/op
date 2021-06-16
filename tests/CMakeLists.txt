
set(dependencies mpi op_core)

if (ENABLE_GTEST)

blt_list_append(TO dependencies ELEMENTS gtest)

blt_add_executable( NAME TwoCnsts
		    HEADERS ${PROJECT_BINARY_DIR}/include/op_config.hpp
		    SOURCES TwoCnsts.cpp
    		    OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
		    DEPENDS_ON nlopt_so ${dependencies}
		    FOLDER      op/tests )

foreach(ntask RANGE 1 2)
blt_add_test( NAME "TwoCnsts_${ntask}"
	      COMMAND       TwoCnsts
	      NUM_MPI_TASKS ${ntask})
endforeach()

blt_add_executable( NAME VariableMap
		    HEADERS ${PROJECT_BINARY_DIR}/include/op_config.hpp
		    SOURCES VariableMap.cpp
		    OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
		    DEPENDS_ON ${dependencies}
		    FOLDER      op/tests )

foreach(ntask RANGE 1 9)
blt_add_test( NAME "VariableMap_${ntask}"
	      COMMAND       VariableMap
	      NUM_MPI_TASKS ${ntask})
endforeach()


endif() # GTEST
