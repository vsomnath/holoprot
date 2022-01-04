CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )
PROJECT( superpixel CXX )

SET( CMAKE_CXX_STANDARD 11 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

########################################################################
# Macros and additional flags
########################################################################

INCLUDE( CheckCXXCompilerFlag )

FILE( GLOB filenames "cmake/Macros/*.cmake" )
FOREACH( filename ${filenames} )
  INCLUDE( ${filename} )
ENDFOREACH()

########################################################################
# Library setup
########################################################################

SET( PACKAGE_VERSION "0.1.1" )

FIND_PACKAGE( pybind11 REQUIRED )
FIND_PACKAGE( Python3 COMPONENTS Interpreter Development NumPy )

IF( Python3_Interpreter_FOUND AND Python3_Development_FOUND AND PYBIND11_FOUND AND PYTHON3_FOUND AND Python3_NumPy_FOUND)

  INCLUDE_DIRECTORIES(SYSTEM ${PYBIND11_INCLUDE_DIRS}
	  		     ${Python3_INCLUDE_DIRS}
			     ${Python3_NumPy_INCLUDE_DIRS})

  ADD_LIBRARY( ers SHARED
      MERCCInput.cpp
      MERCDisjointSet.cpp
      MERCFunctions.cpp
      MERCLazyGreedy.cpp
      MERCOutput.cpp
      ers.cpp
  )

  # The library must not have any prefix and should be located in
  # a subfolder that includes the package name. The setup will be
  # more complicated otherwise.
  SET_TARGET_PROPERTIES( ers
    PROPERTIES
      PREFIX ""
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ers"
  )

  # This is required for linking the library under Mac OS X. Moreover,
  # the suffix ensures that the module can be found by the interpreter
  # later on.
  IF( APPLE )
    SET_TARGET_PROPERTIES( ers
      PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
        SUFFIX     ".so"
    )
  ENDIF()

    # Place the initialization file in the output directory for the Python
    # bindings. This will simplify the installation.
    CONFIGURE_FILE( ers/__init__.py
      ${CMAKE_CURRENT_BINARY_DIR}/ers/__init__.py
    )

    # Ditto for the setup file.
    CONFIGURE_FILE( ers/setup.py
      ${CMAKE_CURRENT_BINARY_DIR}/ers/setup.py
    )
ENDIF()
