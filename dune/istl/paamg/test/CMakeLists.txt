find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  add_executable(pthreadamgtest pthreadamgtest.cc)
  target_compile_definitions(pthreadamgtest PUBLIC MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>)
  target_link_libraries(pthreadamgtest ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  dune_add_test(TARGET pthreadamgtest)

  add_executable(pthreadfastamgtest pthreadamgtest.cc)
  target_compile_definitions(pthreadfastamgtest PUBLIC MYAMG=Dune::Amg::FastAMG<Operator,Vector>)
  add_dune_mpi_flags(pthreadfastamgtest)
  target_link_libraries(pthreadfastamgtest ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  dune_add_test(TARGET pthreadfastamgtest)

  add_executable(pthreadtwoleveltest pthreadtwoleveltest.cc)
  target_link_libraries(pthreadtwoleveltest ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  dune_add_test(TARGET pthreadtwoleveltest)

  if(SUPERLU_FOUND OR SuiteSparse_UMFPACK_FOUND)
    dune_add_test(NAME pthreaddirectamgtest
                  SOURCES pthreadamgtest.cc
                  COMPILE_DEFINITIONS MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>
                  LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

    dune_add_test(NAME pthreaddirectfastamgtest
                  SOURCES pthreadamgtest.cc
                  COMPILE_DEFINITIONS MYAMG=Dune::Amg::FastAMG<Operator,Vector>
                  LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

    dune_add_test(NAME pthreaddirecttwoleveltest
                  SOURCES pthreadtwoleveltest.cc
                  LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  endif()
endif()

# add an executable without SuperLU/UMFPack
add_executable(amgtest amgtest.cc)
target_link_libraries(amgtest ${DUNE_LIBS})
dune_add_test(TARGET amgtest)

add_executable(fastamg fastamg.cc)
target_link_libraries(fastamg ${DUNE_LIBS})
dune_add_test(TARGET fastamg)

if(SUPERLU_FOUND)
  add_executable(superluamgtest amgtest.cc)
  add_dune_superlu_flags(superluamgtest)
  target_link_libraries(superluamgtest ${DUNE_LIBS})
  dune_add_test(TARGET superluamgtest)

  add_executable(superlufastamgtest fastamg.cc)
  add_dune_superlu_flags(superlufastamgtest)
  target_link_libraries(superlufastamgtest ${DUNE_LIBS})
  dune_add_test(TARGET superlufastamgtest)
endif(SUPERLU_FOUND)

if(SuiteSparse_UMFPACK_FOUND)
  dune_add_test(NAME umfpackamgtest
                SOURCES amgtest.cc)

  dune_add_test(NAME umfpackfastamgtest
                SOURCES fastamg.cc)
endif()


dune_add_test(SOURCES twolevelmethodtest.cc)

dune_add_test(SOURCES galerkintest.cc)
dune_add_test(SOURCES graphtest.cc)
dune_add_test(SOURCES hierarchytest.cc)
dune_add_test(SOURCES kamgtest.cc)

dune_add_test(NAME pamg_comm_repart_test
              SOURCES parallelamgtest.cc
              COMPILE_DEFINITIONS -DAMG_REPART_ON_COMM_GRAPH)

dune_add_test(NAME pamgtest
              SOURCES parallelamgtest.cc)

dune_add_test(SOURCES transfertest.cc)

dune_add_test(NAME twolevelmethodschwarztest
              SOURCES twolevelmethodtest.cc
              COMPILE_DEFINITIONS USE_OVERLAPPINGSCHWARZ)
