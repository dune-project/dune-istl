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

function(add_dune_superlu_flags)
  if(SUPERLU_FOUND)
    cmake_parse_arguments(_add_superlu "OBJECT" "" "" ${ARGN})
    foreach(_target ${_add_superlu_UNPARSED_ARGUMENTS})
      if(NOT _add_superlu_OBJECT)
        target_link_libraries(${_target} ${SUPERLU_DUNE_LIBRARIES})
      endif()
      get_target_property(_props ${_target} COMPILE_FLAGS)
      string(REPLACE "_props-NOTFOUND" "" _props "${_props}")
      set_target_properties(${_target} PROPERTIES COMPILE_FLAGS
        "${_props} ${SUPERLU_DUNE_COMPILE_FLAGS} -DENABLE_SUPERLU=1")
    endforeach()
  endif(SUPERLU_FOUND)
endfunction(add_dune_superlu_flags)

