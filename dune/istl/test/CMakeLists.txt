# install the test tools as we want to support testing 3rd-party vectors and matrices with an installed dune-istl
install(FILES
        matrixtest.hh
        vectortest.hh
        laplacian.hh
        multirhstest.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/test)

dune_add_test(SOURCES bcrsassigntest.cc)

dune_add_test(SOURCES bcrsmatrixtest.cc)

dune_add_test(SOURCES bcrsnormtest.cc)

dune_add_test(SOURCES cgconditiontest.cc)

dune_add_test(SOURCES dotproducttest.cc)

dune_add_test(SOURCES complexmatrixtest.cc)

dune_add_test(SOURCES fieldvectortest.cc)

dune_add_test(SOURCES matrixnormtest.cc)

dune_add_test(SOURCES matrixutilstest.cc)

dune_add_test(SOURCES matrixtest.cc)

dune_add_test(SOURCES multirhstest.cc)
  add_dune_vc_flags(multirhstest)

dune_add_test(SOURCES bvectortest.cc)

dune_add_test(SOURCES vbvectortest.cc)

dune_add_test(SOURCES bcrsbuild.cc)

dune_add_test(SOURCES bcrsimplicitbuild.cc
              COMPILE_DEFINITIONS DUNE_ISTL_WITH_CHECKING=1)

dune_add_test(SOURCES matrixiteratortest.cc)

dune_add_test(SOURCES mmtest.cc)

dune_add_test(SOURCES multitypeblockmatrixtest.cc)

dune_add_test(SOURCES multitypeblockvectortest.cc)

dune_add_test(SOURCES mv.cc)

dune_add_test(SOURCES iotest.cc)

dune_add_test(SOURCES inverseoperator2prectest.cc)

dune_add_test(SOURCES preconditionerstest.cc)

dune_add_test(SOURCES scalarproductstest.cc)

dune_add_test(SOURCES scaledidmatrixtest.cc)

dune_add_test(SOURCES solvertest.cc)

dune_add_test(SOURCES solveraborttest.cc)

dune_add_test(SOURCES solverrepositorytest.cc
                LINK_LIBRARIES "-Wl,-whole-archive" duneistl "-Wl,-no-whole-archive")

dune_symlink_to_source_files(FILES solverrepositorytest.ini)

# SuperLU tests
  dune_add_test(NAME superlustest
                SOURCES superlutest.cc
                COMPILE_DEFINITIONS SUPERLU_NTYPE=0)

  dune_add_test(SOURCES superlutest.cc)

  dune_add_test(NAME superluctest
                SOURCES superlutest.cc
                COMPILE_DEFINITIONS SUPERLU_NTYPE=2)

  dune_add_test(NAME superluztest
                SOURCES superlutest.cc
                COMPILE_DEFINITIONS SUPERLU_NTYPE=3)

  dune_add_test(SOURCES complexrhstest.cc
                COMPILE_DEFINITIONS SUPERLU_NTYPE=3)

# SuiteSparse tests
  dune_add_test(SOURCES umfpacktest.cc)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "umfpack_decomp")

dune_add_test(SOURCES ldltest.cc)

dune_add_test(SOURCES spqrtest.cc)

dune_add_test(SOURCES overlappingschwarztest.cc)

dune_add_test(SOURCES cholmodtest.cc)

dune_add_test(SOURCES registrytest.cc)

# MPI tests
dune_add_test(SOURCES matrixredisttest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES vectorcommtest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES matrixmarkettest.cc)

dune_add_test(SOURCES iluildltest.cc)

exclude_from_headercheck(complexdata.hh)
