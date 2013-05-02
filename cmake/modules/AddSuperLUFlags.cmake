#
# Module providing convenience methods for compile binaries with SuperLU support.
#
# Provides the following functions:
#
# add_dune_superlu_flags(target1 target2 ...)
#
# adds SuperLU flags to the targets for compilation and linking
#
function(add_dune_superlu_flags _targets)
  if(SUPERLU_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} ${SUPERLU_DUNE_LIBRARIES})
      get_target_property(_props ${_target} COMPILE_FLAGS)
      string(REPLACE "_props-NOTFOUND" "" _props "${_props}")
      set_target_properties(${_target} PROPERTIES COMPILE_FLAGS
        "${_props} ${SUPERLU_DUNE_COMPILE_FLAGS} -DENABLE_SUPERLU=1")
    endforeach(_target ${_targets})
  endif(SUPERLU_FOUND)
endfunction(add_dune_superlu_flags)

