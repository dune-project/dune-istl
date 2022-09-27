# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Defines the functions to use ARPACKPP
#
# .. cmake_function:: add_dune_arpackpp_flags
#
#    .. cmake_param:: targets
#       :positional:
#       :single:
#       :required:
#
#       A list of targets to use ARPACKPP with.
#

function(add_dune_arpackpp_flags _targets)
  if(ARPACKPP_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} PUBLIC ${ARPACKPP_DUNE_LIBRARIES})
      target_compile_definitions(${_target} PUBLIC ENABLE_ARPACKPP=1)
      target_compile_options(${_target} PUBLIC ${ARPACKPP_DUNE_COMPILE_FLAGS})
    endforeach()
  endif()
endfunction(add_dune_arpackpp_flags)
