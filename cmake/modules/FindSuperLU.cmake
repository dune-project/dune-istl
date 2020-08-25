# .. cmake_module::
#
#    Module that checks whether SuperLU is available and usable.
#    SuperLU must be 5.0 or newer.
#
#    Sets the follwing variables:
#
#    :code:`SUPERLU_FOUND`
#       True if SuperLU available and usable.
#
#    :code:`SUPERLU_INCLUDE_DIRS`
#       Path to the SuperLU include dirs.
#       .. deprecated:: 2.8
#          Use target SuperLU::SuperLU instead.
#
#    :code:`SUPERLU_LIBRARIES`
#       Name to the SuperLU library.
#       .. deprecated:: 2.8
#          Use target SuperLU::SuperLU instead.
#
#
#    This module provides the following imported targets, if found:
#
#    :code:`SuperLU:SuperLU`
#      Library and include directories for the found SuperLU.
#
#
#    This module provides the following imported targets, if found:
#
#    :code:`SuperLU:SuperLU`
#      Library and include directories for the found SuperLU.
#

# text for feature summary
set_package_properties("SuperLU" PROPERTIES
  DESCRIPTION "Supernodal LU"
  PURPOSE "Direct solver for linear system, based on LU decomposition")

find_package(BLAS QUIET)

find_path(SUPERLU_INCLUDE_DIR
  NAMES supermatrix.h
  PATH_SUFFIXES "superlu" "SuperLU" "include/superlu" "include" "SRC"
)

find_library(SUPERLU_LIBRARY
  NAMES "superlu"
        "superlu_5.2.1" "superlu_5.2" "superlu_5.1.1" "superlu_5.1" "superlu_5.0"
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

# check whether version is at least 5.0
check_c_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main(void)
{
  static_assert(SUPERLU_MAJOR_VERSION >= 5, \"SuperLU must be 5.0 or newer.\");
  return 0;
}"
SUPERLU_MIN_VERSION_5)

include(CheckIncludeFiles)
set(HAVE_SLU_DDEFS_H 1)
check_include_files(slu_sdefs.h HAVE_SLU_SDEFS_H)
check_include_files(slu_cdefs.h HAVE_SLU_CDEFS_H)
check_include_files(slu_zdefs.h HAVE_SLU_ZDEFS_H)

cmake_pop_check_state()

set(SUPERLU_INT_TYPE "int" CACHE STRING
  "The integer version that SuperLU was compiled for (Default is int.
  Should be the same as int_t define in e.g. slu_sdefs.h")

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SuperLU"
  DEFAULT_MSG
  SUPERLU_INCLUDE_DIR
  SUPERLU_LIBRARY
  SUPERLU_MIN_VERSION_5
)

mark_as_advanced(SUPERLU_INCLUDE_DIR SUPERLU_LIBRARY SUPERLU_MIN_VERSION_5)


# if both headers and library are found, store results
if(SUPERLU_FOUND)
  if(NOT TARGET SuperLU::SuperLU)
    add_library(SuperLU::SuperLU UNKNOWN IMPORTED)
    set_target_properties(SuperLU::SuperLU
      PROPERTIES
      IMPORTED_LOCATION ${SUPERLU_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES "${SUPERLU_INCLUDE_DIR}")
    # Link BLAS library
    if(TARGET BLAS::BLAS)
      target_link_libraries(SuperLU::SuperLU
        INTERFACE BLAS::BLAS)
    else()
      target_link_libraries(SuperLU::SuperLU
        INTERFACE ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})
    endif()
  endif()
  set(SUPERLU_INCLUDE_DIRS ${SUPERLU_INCLUDE_DIR})
  set(SUPERLU_LIBRARIES    ${SUPERLU_LIBRARY})
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determining location of SuperLU succeeded:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n\n")
  set(SUPERLU_DUNE_COMPILE_FLAGS "-I${SUPERLU_INCLUDE_DIRS}"
    CACHE STRING "Compile flags used by DUNE when compiling SuperLU programs")
  set(SUPERLU_DUNE_LIBRARIES ${SUPERLU_LIBRARIES} ${BLAS_LIBRARIES}
    CACHE STRING "Libraries used by DUNE when linking SuperLU programs")
else()
  # log errornous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining location of SuperLU failed:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIRS}\n"
    "Library directory: ${SUPERLU_LIBRARIES}\n")
endif(SUPERLU_FOUND)
