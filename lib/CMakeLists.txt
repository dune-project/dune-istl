install(FILES duneistlam2cmake.lib
  DESTINATION ${CMAKE_INSTALL_BINDIR}/../lib)

LIST(APPEND DUNE_ISTL_PRECOMPILE_BLOCKS
  "double"
  "std::complex<double>"
  "float"
  "std::complex<float>"
  "Dune::FieldVector<double,1>"
  "Dune::FieldVector<std::complex<double>,1>"
  "Dune::FieldVector<float,1>"
  "Dune::FieldVector<double,2>"
  "Dune::FieldVector<std::complex<double>,2>"
  "Dune::FieldVector<float,2>"
  "Dune::FieldVector<double,3>"
  "Dune::FieldVector<std::complex<double>,3>"
  "Dune::FieldVector<float,3>"
  )

set(COUNTER 0)
foreach(BLOCK ${DUNE_ISTL_PRECOMPILE_BLOCKS})
  configure_file(directsolvers.cc.in directsolvers_${COUNTER}.cc)
  list(APPEND DUNE_ISTL_DIRECTSOLVERS_SOURCES directsolvers_${COUNTER}.cc)
  configure_file(preconditioners.cc.in preconditioners_${COUNTER}.cc)
  list(APPEND DUNE_ISTL_PRECONDITIONERS_SOURCES preconditioners_${COUNTER}.cc)
  configure_file(iterativesolvers.cc.in iterativesolvers_${COUNTER}.cc)
  list(APPEND DUNE_ISTL_ITERATIVESOLVERS_SOURCES iterativesolvers_${COUNTER}.cc)
  MATH(EXPR COUNTER "${COUNTER}+1")
endforeach()

dune_add_library(duneistl
  ${DUNE_ISTL_DIRECTSOLVERS_SOURCES}
  ${DUNE_ISTL_PRECONDITIONERS_SOURCES}
  ${DUNE_ISTL_ITERATIVESOLVERS_SOURCES})
add_dune_all_flags(duneistl)
target_link_libraries(duneistl ${DUNE_LIBS})
