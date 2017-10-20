set(HEADERS
  bcrsmatrix.hh
  bvector.hh
  iterator.hh
  operators.hh
  preconditioners.hh
  slice.hh
  solvers.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/python/istl)
