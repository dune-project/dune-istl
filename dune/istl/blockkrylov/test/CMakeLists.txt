dune_add_test(SOURCES blockcgtest.cc)
dune_add_test(SOURCES blockgmrestest.cc)
dune_add_test(SOURCES blockbicgstabtest.cc)

dune_add_test(SOURCES matrixalgebratest.cc)
dune_add_test(NAME matrixalgebratest_without_blas SOURCES matrixalgebratest.cc)
target_compile_definitions(matrixalgebratest_without_blas PUBLIC TEST_WITHOUT_BLAS=1)

dune_add_test(SOURCES blastest.cc CMAKE_GUARD BLAS_FOUND)

dune_symlink_to_source_files(FILES blockkrylovfactorytest.ini)
dune_add_test(SOURCES blockkrylovfactorytest.cc
  MPI_RANKS 1 2 4
  TIMEOUT 600)

dune_symlink_to_source_files(FILES breakdown_test.ini)
dune_add_test(SOURCES breakdown_test.cc)
