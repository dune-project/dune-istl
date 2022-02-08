find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  add_executable(pthreadamgtest pthreadamgtest.cc)
  target_compile_definitions(pthreadamgtest PUBLIC MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>)
  target_link_libraries(pthreadamgtest PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  add_dune_mpi_flags(pthreadamgtest)
  add_dune_parmetis_flags(pthreadamgtest)
  dune_add_test(TARGET pthreadamgtest)

  add_executable(pthreadfastamgtest pthreadamgtest.cc)
  target_compile_definitions(pthreadfastamgtest PUBLIC MYAMG=Dune::Amg::FastAMG<Operator,Vector>)
  target_link_libraries(pthreadfastamgtest PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  add_dune_mpi_flags(pthreadfastamgtest)
  add_dune_parmetis_flags(pthreadfastamgtest)
  dune_add_test(TARGET pthreadfastamgtest)

  add_executable(pthreadtwoleveltest pthreadtwoleveltest.cc)
  target_link_libraries(pthreadtwoleveltest PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${DUNE_LIBS})
  add_dune_mpi_flags(pthreadtwoleveltest)
  add_dune_parmetis_flags(pthreadtwoleveltest)
  dune_add_test(TARGET pthreadtwoleveltest)

  dune_add_test(NAME pthreaddirectamgtest
                SOURCES pthreadamgtest.cc
                COMPILE_DEFINITIONS MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")

  dune_add_test(NAME pthreaddirectfastamgtest
                SOURCES pthreadamgtest.cc
                COMPILE_DEFINITIONS MYAMG=Dune::Amg::FastAMG<Operator,Vector>
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")

  dune_add_test(NAME pthreaddirecttwoleveltest
                SOURCES pthreadtwoleveltest.cc
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")
endif()

# add an executable without SuperLU/UMFPack
add_executable(amgtest amgtest.cc)
target_link_libraries(amgtest PRIVATE ${DUNE_LIBS})
add_dune_mpi_flags(amgtest)
add_dune_parmetis_flags(amgtest)
dune_add_test(TARGET amgtest)

add_executable(fastamg fastamg.cc)
target_link_libraries(fastamg PRIVATE ${DUNE_LIBS})
add_dune_mpi_flags(fastamg)
add_dune_parmetis_flags(fastamg)
dune_add_test(TARGET fastamg)

if(SuperLU_FOUND)
  add_executable(superluamgtest amgtest.cc)
  target_link_libraries(superluamgtest PRIVATE ${DUNE_LIBS})
  add_dune_superlu_flags(superluamgtest)
  add_dune_mpi_flags(superluamgtest)
  add_dune_parmetis_flags(superluamgtest)
  dune_add_test(TARGET superluamgtest)

  add_executable(superlufastamgtest fastamg.cc)
  target_link_libraries(superlufastamgtest PRIVATE ${DUNE_LIBS})
  add_dune_superlu_flags(superlufastamgtest)
  add_dune_mpi_flags(superlufastamgtest)
  add_dune_parmetis_flags(superlufastamgtest)
  dune_add_test(TARGET superlufastamgtest)
endif()

dune_add_test(NAME umfpackamgtest
              SOURCES amgtest.cc
              CMAKE_GUARD SuiteSparse_UMFPACK_FOUND)
add_dune_mpi_flags(umfpackamgtest)
add_dune_parmetis_flags(umfpackamgtest)

dune_add_test(NAME umfpackfastamgtest
              SOURCES fastamg.cc
              CMAKE_GUARD SuiteSparse_UMFPACK_FOUND)
add_dune_mpi_flags(umfpackfastamgtest)
add_dune_parmetis_flags(umfpackfastamgtest)

dune_add_test(SOURCES twolevelmethodtest.cc)

dune_add_test(SOURCES graphtest.cc)

dune_add_test(SOURCES kamgtest.cc)

dune_add_test(SOURCES transfertest.cc)

dune_add_test(NAME twolevelmethodschwarztest
              SOURCES twolevelmethodtest.cc
              COMPILE_DEFINITIONS USE_OVERLAPPINGSCHWARZ)

# These tests need MPI, even though they are only run sequentially
dune_add_test(SOURCES galerkintest.cc
              CMAKE_GUARD MPI_FOUND)

dune_add_test(SOURCES hierarchytest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

dune_add_test(NAME pamgtest
              SOURCES parallelamgtest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)

dune_add_test(NAME pamg_comm_repart_test
              MPI_RANKS 1 2 4
              TIMEOUT 600
              SOURCES parallelamgtest.cc
              COMPILE_DEFINITIONS -DAMG_REPART_ON_COMM_GRAPH
              CMAKE_GUARD MPI_FOUND)
add_dune_parmetis_flags(pamg_comm_repart_test)

dune_add_test(NAME pamgmmtest
  SOURCES pamgmmtest.cc
  TIMEOUT 20
  MPI_RANKS 6
  CMAKE_GUARD PARMETIS_FOUND)

file(COPY gr_30_30.mm DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
