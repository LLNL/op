blt_add_library( NAME    op_core
                 HEADERS op.hpp op_utility.hpp op_mpi.hpp op_waitloop.hpp
		 DEPENDS_ON dl mpi)

blt_add_library( NAME test_optimizer
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op_core mpi
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


install(FILES op.hpp nlopt_op.hpp op_utility.hpp op_mpi.hpp op_waitloop.hpp ${PROJECT_BINARY_DIR}/include/op_config.hpp DESTINATION include/ )

install(FILES op-config.cmake DESTINATION lib/cmake )

install(TARGETS              op_core nlopt_so
        EXPORT               op
        DESTINATION          lib
        )

set(dependencies mpi op_core)

blt_list_append(TO dependencies ELEMENTS nlopt IF ${USE_NLOPT})

blt_add_executable( NAME demo
		    SOURCES demo.cpp
		    DEPENDS_ON ${dependencies})

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()
