# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  dune_add_test(NAME pthreadamgtest SOURCES pthreadamgtest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
    COMPILE_DEFINITIONS MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>)
  add_dune_mpi_flags(pthreadamgtest)
  add_dune_parmetis_flags(pthreadamgtest)

  dune_add_test(NAME pthreadfastamgtest SOURCES pthreadamgtest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
    COMPILE_DEFINITIONS MYAMG=Dune::Amg::FastAMG<Operator,Vector>)
  add_dune_mpi_flags(pthreadfastamgtest)
  add_dune_parmetis_flags(pthreadfastamgtest)

  dune_add_test(NAME pthreadtwoleveltest SOURCES pthreadtwoleveltest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  add_dune_mpi_flags(pthreadtwoleveltest)
  add_dune_parmetis_flags(pthreadtwoleveltest)

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
dune_add_test(NAME amgtest SOURCES amgtest.cc)
add_dune_mpi_flags(amgtest)
add_dune_parmetis_flags(amgtest)

dune_add_test(NAME fastamg SOURCES fastamg.cc)
add_dune_mpi_flags(fastamg)
add_dune_parmetis_flags(fastamg)

if(SuperLU_FOUND)
  dune_add_test(NAME superluamgtest SOURCES amgtest.cc)
  add_dune_superlu_flags(superluamgtest)
  add_dune_mpi_flags(superluamgtest)
  add_dune_parmetis_flags(superluamgtest)

  dune_add_test(NAME superlufastamgtest SOURCES fastamg.cc)
  add_dune_superlu_flags(superlufastamgtest)
  add_dune_mpi_flags(superlufastamgtest)
  add_dune_parmetis_flags(superlufastamgtest)
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
