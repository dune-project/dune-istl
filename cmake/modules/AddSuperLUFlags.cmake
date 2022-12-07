# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Defines the functions to use SuperLU
#
# .. cmake_function:: add_dune_superlu_flags
#
#    .. cmake_param:: targets
#       :positional:
#       :single:
#       :required:
#
#       A list of targets to use SuperLU with.
#

# set HAVE_SUPERLU for config.h
set(HAVE_SUPERLU ${SuperLU_FOUND})

# register all SuperLU related flags
if(SuperLU_FOUND)
  dune_register_package_flags(
    COMPILE_DEFINITIONS "ENABLE_SUPERLU=1"
    LIBRARIES SuperLU::SuperLU)
endif()

# Provide function to set target properties for linking to SuperLU
function(add_dune_superlu_flags _targets)
  if(SuperLU_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} PUBLIC SuperLU::SuperLU)
      target_compile_definitions(${_target} PUBLIC ENABLE_SUPERLU=1)
    endforeach()
  endif()
endfunction(add_dune_superlu_flags)

