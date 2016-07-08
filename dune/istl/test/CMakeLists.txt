# install the test tools as we want to support testing 3rd-party vectors with an installed dune-istl
install(FILES vectortest.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/test)

dune_add_test(SOURCES basearraytest.cc)

dune_add_test(SOURCES bcrsassigntest.cc)

dune_add_test(SOURCES bcrsnormtest.cc)

dune_add_test(SOURCES dotproducttest.cc)

dune_add_test(SOURCES complexmatrixtest.cc)

dune_add_test(SOURCES fieldvectortest.cc)

dune_add_test(SOURCES factorytest.cc MPI_RANKS 1 2 TIMEOUT 30)
dune_symlink_to_source_files(FILES factorytest_config.ini)

dune_add_test(SOURCES matrixnormtest.cc)

dune_add_test(SOURCES matrixutilstest.cc)

dune_add_test(SOURCES matrixtest.cc)

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

dune_add_test(SOURCES scaledidmatrixtest.cc)

dune_add_test(SOURCES solveraborttest.cc
              SKIP_ON_77)

# Pardiso tests
dune_add_test(SOURCES test_pardiso.cc
              SKIP_ON_77)

# SuperLU tests
dune_add_test(NAME superlustest
              SOURCES superlutest.cc
              COMPILE_DEFINITIONS SUPERLU_NTYPE=0
              SKIP_ON_77)

dune_add_test(SOURCES superlutest.cc
              SKIP_ON_77)

dune_add_test(NAME superluctest
              SOURCES superlutest.cc
              COMPILE_DEFINITIONS SUPERLU_NTYPE=2
              SKIP_ON_77)

dune_add_test(NAME superluztest
              SOURCES superlutest.cc
              COMPILE_DEFINITIONS SUPERLU_NTYPE=3
              SKIP_ON_77)

dune_add_test(SOURCES complexrhstest.cc
              COMPILE_DEFINITIONS SUPERLU_NTYPE=3)

# SuiteSparse tests
dune_add_test(SOURCES umfpacktest.cc
              SKIP_ON_77)
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "umfpack_decomp")

dune_add_test(SOURCES ldltest.cc
              SKIP_ON_77)

dune_add_test(SOURCES spqrtest.cc
              SKIP_ON_77)

dune_add_test(SOURCES overlappingschwarztest.cc
              SKIP_ON_77)

# MPI tests
if(MPI_FOUND)
  dune_add_test(SOURCES matrixredisttest.cc)

  dune_add_test(SOURCES vectorcommtest.cc)
endif()

dune_add_test(SOURCES matrixmarkettest.cc)

exclude_from_headercheck(complexdata.hh)
