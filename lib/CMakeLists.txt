install(FILES duneistlam2cmake.lib
  DESTINATION ${CMAKE_INSTALL_BINDIR}/../lib)

LIST(APPEND DUNE_ISTL_PRECOMPILE_BLOCKS
  "double"
  "Dune::FieldVector<double,1>"
  "std::complex<double>")

set(COUNTER 0)
foreach(BLOCK ${DUNE_ISTL_PRECOMPILE_BLOCKS})
  configure_file(directsolvers.cc.in directsolvers_${COUNTER}.cc)
  list(APPEND DUNE_ISTL_DIRECTSOLVERS_SOURCES directsolvers_${COUNTER}.cc)
  MATH(EXPR COUNTER "${COUNTER}+1")
endforeach()

dune_add_library(duneistl
  ${DUNE_ISTL_DIRECTSOLVERS_SOURCES})
add_dune_all_flags(duneistl)
target_link_libraries(duneistl ${DUNE_LIBS})
