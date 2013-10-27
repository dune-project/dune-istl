set(NORMALTEST
  basearraytest
  bvectortest
  bcrsbuildtest
  dotproducttest
  iotest
  inverseoperator2prectest
  matrixiteratortest
  matrixtest
  matrixutilstest
  mmtest
  mv
  scaledidmatrixtest
  seqmatrixmarkettest
  vbvectortest)

if(HAVE_PARDISO)
  set(PARDISOTEST test_pardiso)
endif(HAVE_PARDISO)

if(SUPERLU_FOUND)
  set(SUPERLUTESTS
  complexrhstest superlutest superluztest superluctest superlustest
  )
endif(SUPERLU_FOUND)

if(UMFPACK_FOUND)
  set(UMFPACKTESTS umfpacktest)
endif()

if ((SUPERLU_FOUND) OR (UMFPACK_FOUND))
  set(OVLPSCHWARZTESTS overlappingschwarztest)
endif()

if(HAVE_MPI)
  set(MPITESTS vectorcommtest matrixmarkettest matrixredisttest)
endif(HAVE_MPI)

set(ALLTESTS ${MPITESTS} ${NORMALTEST} ${PARDISOTEST} ${SUPERLUTESTS} ${UMFPACKTESTS} ${OVLPSCHWARZTESTS})


# We do not want want to build the tests during make all,
# but just build them on demand
add_directory_test_target(_test_target)
add_dependencies(${_test_target} ${ALLTESTS})


include(DuneMPI)

# Provide source files
add_executable(basearraytest "basearraytest.cc")
add_executable(dotproducttest "dotproducttest.cc")
add_executable(matrixutilstest "matrixutilstest.cc")
add_executable(matrixtest "matrixtest.cc")
add_executable(bvectortest "bvectortest.cc")
add_executable(vbvectortest "vbvectortest.cc")
add_executable(bcrsbuildtest "bcrsbuild.cc")
add_executable(matrixiteratortest "matrixiteratortest.cc")
add_executable(mmtest mmtest.cc)
add_executable(mv "mv.cc")
add_executable(iotest "iotest.cc")
add_executable(inverseoperator2prectest "inverseoperator2prectest.cc")
add_executable(scaledidmatrixtest "scaledidmatrixtest.cc")
add_executable(seqmatrixmarkettest "matrixmarkettest.cc")
#set_target_properties(seqmatrixmarkettest PROPERTIES COMPILE_FLAGS
#  "-DMMSEQUENTIAL ${MPI_DUNE_COMPILE_FLAGS} -DENABLE_MPI=1 -DMPICH_SKIP_MPICXX -DMPIPP_H")

if(HAVE_PARDISO)
  add_executable(pardiso_test test_pardiso.cc)

  # TODO add pardiso flags
endif(HAVE_PARDISO)

if(SUPERLU_FOUND)
  add_executable(complexrhstest complexrhstest.cc)
  set_property(TARGET complexrhstest APPEND PROPERTY COMPILE_DEFINITIONS "SUPERLU_NTYPE=3")

  add_executable(superlutest "superlutest.cc")

  add_executable(superlustest "superlutest.cc")
  set_property(TARGET superlustest APPEND PROPERTY COMPILE_DEFINITIONS "SUPERLU_NTYPE=0")

  add_executable(superluctest "superlutest.cc")
  set_property(TARGET superluctest APPEND PROPERTY COMPILE_DEFINITIONS "SUPERLU_NTYPE=2")

  add_executable(superluztest "superlutest.cc")
  set_property(TARGET superluztest APPEND PROPERTY COMPILE_DEFINITIONS "SUPERLU_NTYPE=3")

  add_dune_superlu_flags("${SUPERLUTESTS}")
endif(SUPERLU_FOUND)

if(UMFPACK_FOUND)
  add_executable(umfpacktest umfpacktest.cc)
  add_dune_umfpack_flags(umfpacktest)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "umfpack_decomp")
endif()

if ((SUPERLU_FOUND) OR (UMFPACK_FOUND))
  add_executable(overlappingschwarztest "overlappingschwarztest.cc")
  if (SUPERLU_FOUND)
    add_dune_superlu_flags(overlappingschwarztest)
  endif()
  if (UMFPACK_FOUND)
    add_dune_umfpack_flags(overlappingschwarztest)
  endif()
endif()

if(HAVE_MPI)
  add_executable(matrixredisttest "matrixredisttest.cc")
  add_executable(vectorcommtest "vectorcommtest.cc")
  add_executable(matrixmarkettest "matrixmarkettest.cc")
  add_dune_mpi_flags("${MPITESTS}")
  add_dune_parmetis_flags(matrixredisttest)
endif(HAVE_MPI)

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
  add_test(${_exe} ${_exe})
endforeach(_exe ${ALLTESTS})
