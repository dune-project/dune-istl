#
# Module providing convenience methods for compile binaries with ARPACK++ support.
#
# Provides the following functions:
#
# add_dune_arpackpp_flags(target1 target2 ...)
#
# adds ARPACK++ flags to the targets for compilation and linking
#
function(add_dune_arpackpp_flags _targets)
  if(ARPACKPP_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} ${ARPACKPP_DUNE_LIBRARIES})
      get_target_property(_props ${_target} COMPILE_FLAGS)
      string(REPLACE "_props-NOTFOUND" "" _props "${_props}")
      set_target_properties(${_target} PROPERTIES COMPILE_FLAGS
        "${_props} ${ARPACKPP_DUNE_COMPILE_FLAGS}")
    endforeach(_target ${_targets})
  endif(ARPACKPP_FOUND)
endfunction(add_dune_arpackpp_flags)
