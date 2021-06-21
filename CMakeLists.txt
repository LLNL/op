cmake_minimum_required(VERSION 3.14)
project( OP )
set(ENABLE_MPI ON CACHE BOOL "")
set(ENABLE_ASTYLE      OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------

# _blt_tutorial_include_blt_start
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT git submodule is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Default to C++11 if not set so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/SeracMacros.cmake)

#------------------------------------------------------------------------------
# Setup NLOPT
#------------------------------------------------------------------------------

if (${USE_NLOPT})
blt_find_libraries( FOUND_LIBS NLOPT_LIB
                    NAMES      nlopt
                    REQUIRED   TRUE
                    PATHS      ${NLOPT_DIR}/lib64 ${NLOPT_DIR}/lib)

blt_register_library( NAME                     nlopt
                      INCLUDES                 ${NLOPT_DIR}/include
                      TREAT_INCLUDES_AS_SYSTEM ON
                      LIBRARIES                ${NLOPT_LIB})
message("Found ${NLOPT_LIB}")		    
endif()

# generate config header

set(OP_BUILD_ROOT "${PROJECT_BINARY_DIR}/")

configure_file(
    ${PROJECT_SOURCE_DIR}/src/op_config.hpp.in
    ${PROJECT_BINARY_DIR}/include/op_config.hpp
)


include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(src)

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

op_add_code_checks(PREFIX   op
                   INCLUDES /src/ /tests/
                   EXCLUDES cmake/blt uberenv_libs)

#------------------------------------------------------------------------------
# Export Targets
#------------------------------------------------------------------------------
set(exported_targets
    op_core
    nlopt_so
)

add_library(op INTERFACE)
target_link_libraries(op INTERFACE ${exported_targets})
install(TARGETS              op
        EXPORT               op
        DESTINATION          lib
        )

install(EXPORT op
        NAMESPACE op::
        DESTINATION lib/cmake
        )
