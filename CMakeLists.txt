# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

cmake_minimum_required(VERSION 3.16)
project(dune-istl C CXX)

# guess build tree of dune-common
if(NOT (dune-common_DIR OR dune-common_ROOT OR
        "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${CMAKE_PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})

endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${dune-common_MODULE_PATH}
  "${PROJECT_SOURCE_DIR}/cmake/modules")

#include the dune macros
include(DuneMacros)

# deactivate global include-directories
dune_policy(SET DP_DEFAULT_INCLUDE_DIRS NEW)

# deactivate global calls to add_dune_all_flags in tests
dune_policy(SET DP_TEST_ADD_ALL_FLAGS NEW)

# do not make all suggested dependencies required automatically
dune_policy(SET DP_SUGGESTED_MODULE_DEPENDENCIES_REQUIRED_DOWNSTREAM NEW)

# start a dune project with information from dune.module
dune_project()

# create library target and export it as Dune::ISTL
dune_add_library(duneistl INTERFACE
  EXPORT_NAME ISTL
  LINK_LIBRARIES Dune::Common)

# set include directories to target
dune_default_include_directories(duneistl INTERFACE)

add_subdirectory(cmake/modules)
add_subdirectory(dune)
add_subdirectory(doc)
add_subdirectory(src)

# if Python bindings are enabled, include necessary sub directories.
if( DUNE_ENABLE_PYTHONBINDINGS )
  add_subdirectory(python)
endif()

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project()
