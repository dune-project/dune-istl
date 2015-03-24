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

if(HAVE_PARDISO)
  dune_add_test(SOURCES test_pardiso.cc)
endif(HAVE_PARDISO)

if(SUPERLU_FOUND)
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
endif()

if(SuiteSparse_UMFPACK_FOUND)
  dune_add_test(SOURCES umfpacktest.cc)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "umfpack_decomp")
endif()

dune_add_test(
  SOURCES ldltest.cc
  SKIP_ON_77)

dune_add_test(
  SOURCES spqrtest.cc
  SKIP_ON_77)

if ((SUPERLU_FOUND) OR (SuiteSparse_UMFPACK_FOUND))
  dune_add_test(SOURCES overlappingschwarztest.cc)
endif()

if(HAVE_MPI)
  dune_add_test(SOURCES matrixredisttest.cc)

  dune_add_test(SOURCES vectorcommtest.cc)

  dune_add_test(SOURCES matrixmarkettest.cc)
endif(HAVE_MPI)

exclude_from_headercheck(complexdata.hh)
