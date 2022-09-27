# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# .. cmake_module::
#
#    Module that checks whether ARPACK++ is available and usable.
#
#    Variables used by this module which you may want to set:
#
#    :ref:`ARPACKPP_ROOT`
#       Path list to search for ARPACK++.
#
#    Sets the following variables:
#
#    :code:`ARPACKPP_FOUND`
#       True if ARPACK++ available.
#
#    :code:`ARPACKPP_INCLUDE_DIRS`
#       Path to the ARPACK++ include directories.
#
#    :code:`ARPACKPP_LIBRARIES`
#       Link against these libraries to use ARPACK++.
#
# .. cmake_variable:: ARPACKPP_ROOT
#
#    You may set this variable to have :ref:`FindARPACKPP` look
#    for the ARPACKPP package in the given path before inspecting
#    system paths.
#

# find ARPACK which is required by ARPACK++
find_package(ARPACK)

# look for header files, only at positions given by the user
find_path(ARPACKPP_INCLUDE_DIR
  NAMES "arssym.h"
  PATHS ${ARPACKPP_PREFIX} ${ARPACKPP_ROOT}
  PATH_SUFFIXES "include" "include/arpack++"
  NO_DEFAULT_PATH
)

# look for header files, including default paths
find_path(ARPACKPP_INCLUDE_DIR
  NAMES "arssym.h"
  PATH_SUFFIXES "include" "include/arpack++"
)

# The arpack++ package in Debian also includes a shared library that we have
# to link to. Other versions of arpack++ are header-only.
# Thus we will later use the arpack++ shared library if found and just ignore
# it if it was not found.
find_library(ARPACKPP_LIBRARY
  NAMES "arpack++"
  PATH_SUFFIXES "lib" "lib32" "lib64"
)

# check header usability
include(CMakePushCheckState)
cmake_push_check_state()

# we need if clauses here because variable is set variable-NOTFOUND if the
# searches above were not successful; without them CMake print errors like:
# "CMake Error: The following variables are used in this project, but they
# are set to NOTFOUND. Please set them or make sure they are set and tested
# correctly in the CMake files."
if(ARPACKPP_INCLUDE_DIR)
  set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${ARPACKPP_INCLUDE_DIR})
  if(ARPACKPP_LIBRARY)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
                                 ${ARPACK_LIBRARIES}
                                 ${ARPACKPP_LIBRARY})
  else()
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
                                 ${ARPACK_LIBRARIES})
  endif()
endif()

# end of header usability check
cmake_pop_check_state()

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "ARPACKPP"
  DEFAULT_MSG
  ARPACK_FOUND
  ARPACKPP_INCLUDE_DIR
)

# hide the introduced cmake cached variables in cmake GUIs
mark_as_advanced(ARPACKPP_INCLUDE_DIR ARPACKPP_LIBRARY)

# if headers are found, store results
if(ARPACKPP_FOUND)
  set(ARPACKPP_INCLUDE_DIRS ${ARPACKPP_INCLUDE_DIR})
  if(ARPACKPP_LIBRARY)
    set(ARPACKPP_LIBRARIES ${ARPACKPP_LIBRARY} ${ARPACK_LIBRARIES})
  else(ARPACKPP_LIBRARY)
    set(ARPACKPP_LIBRARIES ${ARPACK_LIBRARIES})
  endif(ARPACKPP_LIBRARY)
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determing location of ARPACK++ succeeded:\n"
    "Include directory: ${ARPACKPP_INCLUDE_DIRS}\n"
    "Libraries to link against: ${ARPACKPP_LIBRARIES}\n\n")

  # the following is a pretty roundabout way of setting include directories, but it's the
  # only way we can force -isystem. And we want the compiler to treat ARPACK++ as a system
  # library to avoid scaring our users with the horrible warnings triggered by the bitrotted
  # ARPACK++ sources.
  #
  # For this to work, only set the COMPILE_OPTIONS (those replaced COMPILE_FLAGS a while ago), never
  # the INCLUDE_DIRECTORIES!
  set(ARPACKPP_DUNE_COMPILE_FLAGS "$<$<BOOL:${ARPACKPP_INCLUDE_DIRS}>:-isystem$<JOIN:${ARPACKPP_INCLUDE_DIRS}, -isystem>>"
    CACHE STRING "Compile flags used by DUNE when compiling ARPACK++ programs")
  set(ARPACKPP_DUNE_LIBRARIES ${ARPACKPP_LIBRARIES}
    CACHE STRING "Libraries used by DUNE when linking ARPACK++ programs")
else()
  # log errornous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
    "Determing location of ARPACK++ failed:\n"
    "Include directory: ${ARPACKPP_INCLUDE_DIRS}\n"
    "Libraries to link against: ${ARPACKPP_LIBRARIES}\n\n")
endif()

# set HAVE_ARPACKPP for config.h
set(HAVE_ARPACKPP ${ARPACKPP_FOUND})

# register all ARPACK++ related flags
if(ARPACKPP_FOUND)
  dune_register_package_flags(COMPILE_DEFINITIONS "ENABLE_ARPACKPP=1"
                              LIBRARIES "${ARPACKPP_LIBRARIES}"
                              COMPILE_OPTIONS "${ARPACKPP_DUNE_COMPILE_FLAGS}")
endif()

# text for feature summary
set_package_properties("ARPACKPP" PROPERTIES
  DESCRIPTION "ARPACK++"
  PURPOSE "C++ interface for ARPACK")
