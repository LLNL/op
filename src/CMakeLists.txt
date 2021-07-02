blt_append_custom_compiler_flag(FLAGS_VAR _flags
				CLANG "-Wno-return-type-c-linkage")

blt_add_library( NAME    op_core
                 HEADERS op.hpp op_utility.hpp op_mpi.hpp op_waitloop.hpp op_debug.hpp
		 DEPENDS_ON dl mpi)
if (_flags)
set_source_files_properties(test_optimizer.cpp PROPERTIES COMPILE_FLAGS ${_flags})
endif()
blt_add_library( NAME test_optimizer
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op_core mpi
		 SHARED TRUE)

target_include_directories(op_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

if (NLOPT_DIR)
if (_flags)
set_source_files_properties(nlopt_op.cpp PROPERTIES COMPILE_FLAGS ${_flags})
endif()
blt_add_library( NAME nlopt_so
		 HEADERS nlopt_op.hpp
		 SOURCES nlopt_op.cpp
		 DEPENDS_ON op_core nlopt
		 SHARED TRUE)

target_include_directories(nlopt_so INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

endif()

install(FILES op.hpp nlopt_op.hpp op_utility.hpp op_mpi.hpp op_waitloop.hpp ${CMAKE_BINARY_DIR}/include/op/op_config.hpp DESTINATION include/op/ )

install(FILES op-config.cmake DESTINATION lib/cmake )

set(op_targets op_core)
blt_list_append(TO op_targets ELEMENTS nlopt_so IF NLOPT_DIR)
install(TARGETS              ${op_targets}
        EXPORT               op
        DESTINATION          lib
        )

set(dependencies mpi ${op_targets})

blt_add_executable( NAME demo
		    SOURCES demo.cpp
		    DEPENDS_ON ${dependencies})

if (ENABLE_DOCS)
    add_subdirectory(docs)
endif()
