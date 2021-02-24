blt_add_library( NAME    op
                 HEADERS op.hpp 
                 SOURCES op.cpp
		 DEPENDS_ON dl)

blt_add_library( NAME test_optimizer
		 HEADERS op.hpp
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op
		 SHARED TRUE)

blt_add_library( NAME nlopt_so
		 HEADERS op.hpp nlopt.hpp
		 SOURCES nlopt.cpp
		 DEPENDS_ON op nlopt
		 SHARED TRUE)

if (${USE_NLOPT})
set(DEPENDENCIES mpi gtest nlopt ) 
endif()


blt_add_executable( NAME demo
		    HEADERS op.hpp 
		    SOURCES demo.cpp
		    DEPENDS_ON op ${DEPENDENCIES})

blt_add_executable( NAME TwoCnsts
		    HEADERS op.hpp ${PROJECT_BINARY_DIR}/include/op_config.hpp
		    SOURCES TwoCnsts.cpp 
		    DEPENDS_ON op nlopt_so ${DEPENDENCIES})

