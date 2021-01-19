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

if(ARPACKPP_FOUND)
  dune_generate_pkg_config("arpack"
    NAME "ARPACK"
    DESCRIPTION "ARnoldi PACKage"
    URL "https://www.caam.rice.edu/software/ARPACK"
    # CFLAGS "-I${ARPACK_INCLUDE_DIR}"
    LIBS "${ARPACK_LIBRARY}")
    dune_generate_pkg_config("arpackpp"
    NAME "ARPACK++"
    DESCRIPTION "ARnoldi PACKage C++ interface"
    URL "https://github.com/m-reuter/arpackpp"
    CFLAGS "-I${ARPACKPP_INCLUDE_DIR}"
    LIBS "${ARPACKPP_LIBRARY}"
    REQUIRES "arpack")
  dune_add_pkg_config_requirement("arpackpp")
  dune_add_pkg_config_flags("-DHAVE_ARPACKPP")
endif()

function(add_dune_arpackpp_flags _targets)
  if(ARPACKPP_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} PUBLIC ${ARPACKPP_DUNE_LIBRARIES})
      target_compile_definitions(${_target} PUBLIC ENABLE_ARPACKPP=1)
      target_compile_options(${_target} PUBLIC ${ARPACKPP_DUNE_COMPILE_FLAGS})
    endforeach()
  endif()
endfunction(add_dune_arpackpp_flags)
