set(MPITESTS vectorcommtest matrixmarkettestmatrixredisttest )
set(NORMALTEST basearraytest matrixutilstest matrixtest
  bvectortest vbvectortest bcrsbuildtest matrixiteratortest mv
  iotest scaledidmatrixtest seqmatrixmarkettest)

if(HAVE_PARDISO)
  set(PARDISOTEST test_pardiso)
endif(HAVE_PARDISO)

if(HAVE_SUPERLU)
  set(SUPERLUTESTS superlutest superluztest superluctest superlustest
    overlappingschwarztest)
endif(HAVE_SUPERLU)

set(NORMALTEST amgtest graphtest kamgtest)

# We do not want want to build the tests during make all,
# but just build them on demand
add_dependencies(build_tests ${MPITESTS} ${NORMALTEST} ${PARDISOTEST} ${SUPERLUTESTS})

# Provide source files
if(HAVE_PARDISO)
  add_executable(pardiso_test test_pardiso.cc)

  # TODO add pardiso flags
endif(HAVE_PARDISO)

if(HAVE_SUPERLU)
  add_executable(superlutest "superlutest.cc")

  add_executable(superlustest "superlutest.cc")
  set_target_properties(superlustest PROPERTIES COMPILE_FLAGS "-DSUPERLU_NTYPE=0")

  add_executable(superluctest "superlutest.cc")
  set_target_properties(superlustest PROPERTIES COMPILE_FLAGS "-DSUPERLU_NTYPE=2")

  add_executable(superluztest "superlutest.cc")
  set_target_properties(superlustest PROPERTIES COMPILE_FLAGS "-DSUPERLU_NTYPE=3")

  add_executable(overlappingschwarztest "overlappingschwarztest.cc")

  # TODO add superlu flags
endif(HAVE_SUPERLU)

include(DuneMPI)

if(HAVE_MPI)
  add_executable(matrixredisttest "matrixredisttest.cc")
  add_executable(vectorcommtest "vectorcommtest.cc")
  add_executable(matrixmarkettest "matrixmarkettest.cc")
  add_dune_mpi_flags(${MPITESTS})
endif(HAVE_MPI)

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
endforeach(_exe ${ALLTESTS})
