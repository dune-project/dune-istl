# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# .. cmake_module::
#
#    Module that checks whether SuperLU is available and usable.
#
#    Sets the following variables:
#
#    :code:`SuperLU_FOUND`
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
include(FeatureSummary)
set_package_properties("SuperLU" PROPERTIES
  DESCRIPTION "Supernodal LU"
  PURPOSE "Direct solver for linear system, based on LU decomposition")

set(SUPERLU_INT_TYPE "int" CACHE STRING
  "The integer version that SuperLU was compiled for (Default is int.
  Should be the same as int_t define in e.g. slu_sdefs.h")

find_package(BLAS QUIET)

find_path(SUPERLU_INCLUDE_DIR supermatrix.h
  PATH_SUFFIXES "superlu" "SuperLU" "SRC"
)

find_library(SUPERLU_LIBRARY
  NAMES "superlu"
        "superlu_5.2.1" "superlu_5.2" "superlu_5.1.1" "superlu_5.1" "superlu_5.0"
)

# check version of SuperLU
find_file(SLU_UTIL_HEADER slu_util.h
  HINTS ${SUPERLU_INCLUDE_DIR}
  NO_DEFAULT_PATH)
if(SLU_UTIL_HEADER)
  file(READ "${SLU_UTIL_HEADER}" superluheader)
  # get version number from defines in header file
  string(REGEX REPLACE ".*#define SUPERLU_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1"
    SUPERLU_MAJOR_VERSION  "${superluheader}")
  string(REGEX REPLACE ".*#define SUPERLU_MINOR_VERSION[ \t]+([0-9]+).*" "\\1"
    SUPERLU_MINOR_VERSION  "${superluheader}")
  string(REGEX REPLACE ".*#define SUPERLU_PATCH_VERSION[ \t]+([0-9]+).*" "\\1"
    SUPERLU_PATCH_VERSION "${superluheader}")
  if(SUPERLU_MAJOR_VERSION GREATER_EQUAL 0)
    set(SuperLU_VERSION "${SUPERLU_MAJOR_VERSION}")
  endif()
  if (SUPERLU_MINOR_VERSION GREATER_EQUAL 0)
    set(SuperLU_VERSION "${SuperLU_VERSION}.${SUPERLU_MINOR_VERSION}")
  endif()
  if (SUPERLU_PATCH_VERSION GREATER_EQUAL 0)
    set(SuperLU_VERSION "${SuperLU_VERSION}.${SUPERLU_PATCH_VERSION}")
  endif()
  # if SUPERLU_MAJOR_VERSION not defined, SuperLU must be version 4 or older
  if(NOT SuperLU_VERSION)
    set(SuperLU_VERSION "4")
  endif()
endif()

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args("SuperLU"
  REQUIRED_VARS
    SUPERLU_LIBRARY SUPERLU_INCLUDE_DIR SLU_UTIL_HEADER BLAS_FOUND
  VERSION_VAR
    SuperLU_VERSION
)

mark_as_advanced(SUPERLU_INCLUDE_DIR SUPERLU_LIBRARY SLU_UTIL_HEADER)


# if both headers and library are found, store results
if(SuperLU_FOUND)
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
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determining location of SuperLU succeeded:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIR}\n"
    "Library directory: ${SUPERLU_LIBRARY}\n\n")
else()
  # log erroneous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determining location of SuperLU failed:\n"
    "Include directory: ${SUPERLU_INCLUDE_DIR}\n"
    "Library directory: ${SUPERLU_LIBRARY}\n")
endif()
