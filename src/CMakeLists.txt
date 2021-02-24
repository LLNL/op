blt_add_library( NAME    op
                 HEADERS op.hpp
                 SOURCES op.cpp
		 DEPENDS_ON dl)

blt_add_library( NAME test_optimizer
		 HEADERS op.hpp
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op
		 SHARED TRUE)

set(EXECUTABLES
    demo
    TwoCnsts
)

if (${USE_NLOPT})
set(DEPENDENCIES mpi gtest nlopt ) 
endif()

foreach(exe ${EXECUTABLES})
blt_add_executable( NAME ${exe}
		    HEADERS op.hpp
		    SOURCES ${exe}.cpp
		    DEPENDS_ON op ${DEPENDENCIES})
endforeach()
