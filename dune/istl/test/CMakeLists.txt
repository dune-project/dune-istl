dune_add_test(SOURCES basearraytest.cc)

dune_add_test(SOURCES bcrsassigntest.cc)

dune_add_test(SOURCES bcrsnormtest.cc)

dune_add_test(SOURCES dotproducttest.cc)

dune_add_test(SOURCES complexmatrixtest.cc)

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
dune_add_test(SOURCES matrixredisttest.cc)

dune_add_test(SOURCES vectorcommtest.cc)

dune_add_test(SOURCES matrixmarkettest.cc)

exclude_from_headercheck(complexdata.hh)
