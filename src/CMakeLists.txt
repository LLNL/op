blt_add_library( NAME    op_core
                 HEADERS op.hpp 
		 DEPENDS_ON dl)

blt_add_library( NAME test_optimizer
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op_core
		 SHARED TRUE)

blt_add_library( NAME nlopt_so
		 HEADERS nlopt_op.hpp
		 SOURCES nlopt_op.cpp
		 DEPENDS_ON op_core nlopt
		 SHARED TRUE)

target_include_directories(op_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

target_include_directories(nlopt_so INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )


install(FILES op.hpp nlopt_op.hpp DESTINATION include/ )

install(FILES op-config.cmake DESTINATION lib/cmake )

install(TARGETS              op_core nlopt_so
        EXPORT               op
        DESTINATION          lib
        )

if (${USE_NLOPT})
set(DEPENDENCIES mpi gtest nlopt ) 
endif()


blt_add_executable( NAME demo
		    SOURCES demo.cpp
		    DEPENDS_ON op_core ${DEPENDENCIES})

blt_add_executable( NAME TwoCnsts
		    HEADERS ${PROJECT_BINARY_DIR}/include/op_config.hpp
		    SOURCES TwoCnsts.cpp 
		    DEPENDS_ON op_core nlopt_so ${DEPENDENCIES})
