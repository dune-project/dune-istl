# .. cmake_module::
#
#    This modules content is executed whenever a module required or suggests dune-istl!
#
# .. cmake_function:: dune_add_istl_library
#
#    .. cmake_brief::
#
#       Adds a library with precompiled preconditioners and solvers
#
#    .. cmake_param:: BLOCKSIZES
#       :multi:
#
#       The block sizes of vectors/matrices used in your module
#
#
#    This function adds a library to the user's dune module which contains precompiled
#    preconditioners and solvers. This allows for faster compilation of the actual module.
#    This pays off especially when using a factory to create preconditioners and solvers
#    dynamically.
#    The vector/matrix block sizes used in your module must be passed in order for the
#    precompiled preconditioners and solvers to be used.


find_package(METIS)
find_package(ParMETIS)
include(AddParMETISFlags)
find_package(SuperLU)
include(AddSuperLUFlags)
find_package(ARPACKPP)
include(AddARPACKPPFlags)
find_package(SuiteSparse OPTIONAL_COMPONENTS LDL SPQR UMFPACK)
include(AddSuiteSparseFlags)

# enable / disable backwards compatibility w.r.t. category
set(DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE 1
  CACHE BOOL "Enable/Disable the backwards compatibility of the category enum/method in dune-istl solvers, preconditioner, etc. '1'")

# We always have to write the header because it will always be included
file(WRITE ${CMAKE_BINARY_DIR}/solvertemplates.hh "// Placeholder for istl library includes\n")

set (DUNE_ADD_ISTL_LIBRARY_CALLED 0 CACHE INTERNAL "")


# Allow including a library for precompiled preconditioners / solvers
function(dune_add_istl_library)

  cmake_minimum_required(VERSION 3.1)

  # Only allow this to be called once
  if (DUNE_ADD_ISTL_LIBRARY_CALLED)
    message(FATAL_ERROR "You may only call dune_add_istl_library once per module!")
  endif()
  set (DUNE_ADD_ISTL_LIBRARY_CALLED 1 CACHE INTERNAL "")

  # Set library name from current module name
  string(REGEX REPLACE "[^a-zA-Z0-9-]" "_" safe_proj_name ${CMAKE_PROJECT_NAME})
  set(libname ${safe_proj_name}-istl)

  # Check if listed in dune_enable_all_packages
  set (found -1)
  if (DUNE_ENABLE_ALL_PACKAGES_MODULE_LIBRARIES)
    foreach(module_lib ${DUNE_ENABLE_ALL_PACKAGES_MODULE_LIBRARIES})
      if (${module_lib} STREQUAL ${libname})
        set (found 1)
        break()
      endif()
    endforeach()
  endif()
  if (${found} EQUAL -1)
    message(FATAL_ERROR "You have to register ${libname} as a module library in dune_enable_all_packages before calling dune_add_istl_library!")
  endif()



  # Parse input
  include(CMakeParseArguments)
  set(MULTIARGS BLOCKSIZES)
  cmake_parse_arguments(ADDLIB "" "" "${MULTIARGS}" ${ARGN})

  # Check whether the parser produced any errors
  if(ADDLIB_UNPARSED_ARGUMENTS)
    message(WARNING "Unrecognized arguments ('${ADDLIB_UNPARSED_ARGUMENTS}') for dune_add_istl_library!")
  endif()

  # Check BLOCKSIZES for correctness
  if(NOT ADDLIB_BLOCKSIZES)
    message(FATAL_ERROR "BLOCKSIZES has to be specified!")
  endif()
  foreach(bs ${ADDLIB_BLOCKSIZES})
    if(NOT "${bs}" MATCHES "[1-9][0-9]*")
      message(FATAL_ERROR "${bs} was given to the BLOCKSIZES arugment of dune_add_istl_library, but it does not seem like a correct block size number")
    endif()
  endforeach()



  # Get module path of istl
  dune_module_path(MODULE dune-istl RESULT moddir CMAKE_MODULES)

  # Generate template instantiation headers and cc's
  file(WRITE ${CMAKE_BINARY_DIR}/solvertemplates.cc "// Auto-generated template stuff\n")
  file(WRITE ${CMAKE_BINARY_DIR}/solvertemplates.hh "// Auto-generated template stuff\n")
  foreach(bs ${ADDLIB_BLOCKSIZES})
    set(BLOCKSIZE ${bs})
    configure_file("${moddir}/solvertemplates.hh" ${CMAKE_BINARY_DIR}/solvertemplates${bs}.hh)
    configure_file("${moddir}/solvertemplates.cc" ${CMAKE_BINARY_DIR}/solvertemplates${bs}.cc)

    file(APPEND ${CMAKE_BINARY_DIR}/solvertemplates.cc "#include \"solvertemplates${bs}.cc\"\n")
    file(APPEND ${CMAKE_BINARY_DIR}/solvertemplates.hh "#include \"solvertemplates${bs}.hh\"\n")

  endforeach()
  include_directories(${CMAKE_BINARY_DIR})



  # Build library
  dune_library_add_sources(${libname} SOURCES ${CMAKE_BINARY_DIR}/solvertemplates.cc)


endfunction()
