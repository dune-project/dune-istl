# .. cmake_module::
#
#    Module that checks whether SuperLU is available and usable.
#    SuperLU must be 4.0 or newer.
#
#    Variables used by this module which you may want to set:
#
#    :ref:`SUPERLU_ROOT`
#       Path list to search for SuperLU
#
#    Sets the follwing variables:
#
#    :code:`SUPERLU_FOUND`
#       True if SuperLU available and usable.
#
#    :code:`SUPERLU_MIN_VERSION_4`
#       True if SuperLU version >= 4.0.
#
#    :code:`SUPERLU_MIN_VERSION_4_3`
#       True if SuperLU version >= 4.3.
#
#    :code:`SUPERLU_MIN_VERSION_5`
#       True if SuperLU version >= 5.0.
#
#    :code:`SUPERLU_WITH_VERSION`
#       Human readable string containing version information.
#
#    :code:`SUPERLU_INCLUDE_DIRS`
#       Path to the SuperLU include dirs.
#
#    :code:`SUPERLU_LIBRARIES`
#       Name to the SuperLU library.
#
# .. cmake_variable:: SUPERLU_ROOT
#
#    You may set this variable to have :ref:`FindSuperLU` look
#    for the SuperLU package in the given path before inspecting
#    system paths.
#

find_package(BLAS QUIET)

# look for header files, only at positions given by the user
find_path(SUPERLU_INCLUDE_DIR
  NAMES supermatrix.h
  PATHS ${SUPERLU_PREFIX} ${SUPERLU_ROOT}
  PATH_SUFFIXES "superlu" "SuperLU" "include/superlu" "include" "SRC"
  NO_DEFAULT_PATH
)

# look for header files, including default paths
find_path(SUPERLU_INCLUDE_DIR
  NAMES supermatrix.h
  PATH_SUFFIXES "superlu" "SuperLU" "include/superlu" "include" "SRC"
)

# look for library, only at positions given by the user
find_library(SUPERLU_LIBRARY
  NAMES "superlu"
        "superlu_5.2.1" "superlu_5.2" "superlu_5.1.1" "superlu_5.1" "superlu_5.0"
        "superlu_4.3" "superlu_4.2" "superlu_4.1" "superlu_4.0"
  PATHS ${SUPERLU_PREFIX} ${SUPERLU_ROOT}
  PATH_SUFFIXES "lib" "lib32" "lib64"
  NO_DEFAULT_PATH
)

# look for library files, including default paths
find_library(SUPERLU_LIBRARY
  NAMES "superlu"
        "superlu_5.2.1" "superlu_5.2" "superlu_5.1.1" "superlu_5.1" "superlu_5.0"
        "superlu_4.3" "superlu_4.2" "superlu_4.1" "superlu_4.0"
  PATH_SUFFIXES "lib" "lib32" "lib64"
)

# check version specific macros
include(CheckCSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()

# we need if clauses here because variable is set variable-NOTFOUND
# if the searches above were not successful
# Without them CMake print errors like:
# "CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
# Please set them or make sure they are set and tested correctly in the CMake files:"
#
if(SUPERLU_INCLUDE_DIR)
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${SUPERLU_INCLUDE_DIR})
endif(SUPERLU_INCLUDE_DIR)
if(SUPERLU_LIBRARY)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${SUPERLU_LIBRARY})
endif(SUPERLU_LIBRARY)
if(BLAS_LIBRARIES)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${BLAS_LIBRARIES})
endif(BLAS_LIBRARIES)
# check wether version is new enough >= 4.0
check_c_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main()
{
  SuperLUStat_t stat;
  stat.expansions=8;
  return 0;
}" SUPERLU_MIN_VERSION_4)

# check whether version is at least 4.3
check_c_source_compiles("
#include <slu_ddefs.h>
int main(void)
{
  return SLU_DOUBLE;
}"
SUPERLU_MIN_VERSION_4_3)

# check whether version is at least 5.0
check_c_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main(void)
{
  GlobalLU_t glu;
  return 0;
}"
SUPERLU_MIN_VERSION_5)

cmake_pop_check_state()

if(NOT SUPERLU_MIN_VERSION_4)
  set(SUPERLU_WITH_VERSION "SuperLU < 4.0" CACHE STRING
    "Human readable string containing SuperLU version information.")
else()
  if(SUPERLU_MIN_VERSION_5)
    set(SUPERLU_WITH_VERSION "SuperLU >= 5.0" CACHE STRING
      "Human readable string containing SuperLU version information.")
  elseif(SUPERLU_MIN_VERSION_4_3)
    set(SUPERLU_WITH_VERSION "SuperLU >= 4.3" CACHE STRING
      "Human readable string containing SuperLU version information.")
  else()
    set(SUPERLU_WITH_VERSION "SuperLU <= 4.2 and >= 4.0" CACHE STRING
      "Human readable string containing SuperLU version information.")
  endif()
endif()

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SuperLU"
  DEFAULT_MSG
  BLAS_FOUND
  SUPERLU_INCLUDE_DIR
  SUPERLU_LIBRARY
  SUPERLU_MIN_VERSION_4
)

mark_as_advanced(SUPERLU_INCLUDE_DIR SUPERLU_LIBRARY)

set_package_info("SuperLU" "Direct linear solver library")

# if both headers and library are found, store results
if(SUPERLU_FOUND)
  set(SUPERLU_INCLUDE_DIRS ${SUPERLU_INCLUDE_DIR})
  set(SUPERLU_LIBRARIES    ${SUPERLU_LIBRARY})
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determining location of ${SUPERLU_WITH_VERSION} succeeded:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n\n")
  set(SUPERLU_DUNE_COMPILE_FLAGS "-I${SUPERLU_INCLUDE_DIRS}"
    CACHE STRING "Compile flags used by DUNE when compiling SuperLU programs")
  set(SUPERLU_DUNE_LIBRARIES ${SUPERLU_LIBRARIES} ${BLAS_LIBRARIES}
    CACHE STRING "Libraries used by DUNE when linking SuperLU programs")
else(SUPERLU_FOUND)
  # log errornous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining location of SuperLU failed:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n"
    "Found unsupported version: ${SUPERLU_WITH_VERSION}\n\n")
endif(SUPERLU_FOUND)

# set HAVE_SUPERLU for config.h
set(HAVE_SUPERLU ${SUPERLU_FOUND})

# register all superlu related flags
if(SUPERLU_FOUND)
  dune_register_package_flags(COMPILE_DEFINITIONS "ENABLE_SUPERLU=1"
                              LIBRARIES "${SUPERLU_DUNE_LIBRARIES}"
                              INCLUDE_DIRS "${SUPERLU_INCLUDE_DIRS}")
endif()
