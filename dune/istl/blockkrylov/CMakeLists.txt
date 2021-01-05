
add_subdirectory("test")

install(FILES
  blas.hh
  blockcg.hh
  blockinnerproduct.hh
  matrixalgebra.hh
  utils.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/blockkrylov)
