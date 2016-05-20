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
      target_link_libraries(${_target} ${ARPACKPP_DUNE_LIBRARIES})
      get_target_property(_props ${_target} COMPILE_FLAGS)
      string(REPLACE "_props-NOTFOUND" "" _props "${_props}")
      set_target_properties(${_target} PROPERTIES COMPILE_FLAGS
        "${_props} ${ARPACKPP_DUNE_COMPILE_FLAGS} -DENABLE_ARPACKPP=1")
    endforeach()
  endif()
endfunction(add_dune_arpackpp_flags)
