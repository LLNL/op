blt_add_library( NAME    op
                 HEADERS op.hpp
                 SOURCES op.cpp
		 DEPENDS_ON dl)

blt_add_library( NAME test_optimizer
		 HEADERS op.hpp
		 SOURCES test_optimizer.cpp
		 DEPENDS_ON op
		 SHARED TRUE)

blt_add_executable( NAME demo
		    HEADERS op.hpp
		    SOURCES demo.cpp
		    DEPENDS_ON op)
