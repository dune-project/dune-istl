if(MPI_FOUND)
  set(MPITESTS  galerkintest hierarchytest transfertest)
endif(MPI_FOUND)

if(PARMETIS_FOUND)
  set(PARMETISTESTS pamgtest pamg_comm_repart_test)
endif(PARMETIS_FOUND)
if(CMAKE_USE_PTHREADS_INIT)
  set(PTHREADTESTS pthreadamgtest pthreadfastamgtest)
endif(CMAKE_USE_PTHREADS_INIT)
set(NORMALTESTS amgtest fastamg graphtest kamgtest ${PTHREADTESTS})
set(ALLTESTS ${MPITESTS} ${PARMETISTESTS} ${NORMALTESTS})

# We do not want want to build the tests during make all,
# but just build them on demand
add_directory_test_target(_test_target)
add_dependencies(${_test_target} ${ALLTESTS})

find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  # test: pthreadamgtest
  add_executable(pthreadamgtest pthreadamgtest.cc)
  set_property(TARGET pthreadamgtest APPEND PROPERTY
    COMPILE_DEFINITIONS "MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>")
  target_link_libraries(pthreadamgtest "${CMAKE_THREAD_LIBS_INIT}")
  # test: pthreadfastamgtest
  add_executable(pthreadfastamgtest pthreadamgtest.cc)
  set_property(TARGET pthreadfastamgtest APPEND PROPERTY
    COMPILE_DEFINITIONS "MYAMG=Dune::Amg::FastAMG<Operator,Vector>")
  target_link_libraries(pthreadfastamgtest "${CMAKE_THREAD_LIBS_INIT}")
endif(CMAKE_USE_PTHREADS_INIT)

add_executable(amgtest "amgtest.cc")
add_dune_superlu_flags(amgtest)
add_executable(fastamg "fastamg.cc")
add_dune_superlu_flags(fastamg)
add_executable(galerkintest "galerkintest.cc")
add_executable(graphtest "graphtest.cc")
add_executable(hierarchytest "hierarchytest.cc")
add_executable(kamgtest "kamgtest.cc")
add_dune_superlu_flags(kamgtest)
add_executable(pamg_comm_repart_test "parallelamgtest.cc")
set_target_properties(pamg_comm_repart_test PROPERTIES COMPILE_FLAGS "-DAMG_REPART_ON_COMM_GRAPH")
add_executable(pamgtest "parallelamgtest.cc")
add_dune_superlu_flags(pamgtest)
add_executable(transfertest "transfertest.cc")

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
  add_test(${_exe} ${_exe})
endforeach(_exe ${ALLTESTS})

add_dune_mpi_flags("${MPITESTS}")
add_dune_parmetis_flags("${PARMETISTESTS}")
