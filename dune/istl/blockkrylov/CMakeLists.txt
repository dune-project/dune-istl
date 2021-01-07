
set("DUNE_BLOCKKRYLOV_BLAS_PRIORITY" 5 CACHE
  STRING "Set the priority of the BLAS overloads in block Krylov methods")

add_subdirectory("test")

install(FILES
  blas.hh
  blockbicgstab.hh
  blockcg.hh
  blockgmres.hh
  blockinnerproduct.hh
  matrixalgebra.hh
  utils.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/blockkrylov)
