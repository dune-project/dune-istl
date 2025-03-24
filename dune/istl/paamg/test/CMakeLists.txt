# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Link all test targets in this directory against Dune::ISTL
link_libraries(Dune::ISTL)

find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  dune_add_test(NAME pthreadamgtest SOURCES pthreadamgtest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
    COMPILE_DEFINITIONS MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>)

  dune_add_test(NAME pthreadfastamgtest SOURCES pthreadamgtest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
    COMPILE_DEFINITIONS MYAMG=Dune::Amg::FastAMG<Operator,Vector>)

  dune_add_test(NAME pthreadtwoleveltest SOURCES pthreadtwoleveltest.cc
    LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

  dune_add_test(NAME pthreaddirectamgtest
                SOURCES pthreadamgtest.cc
                COMPILE_DEFINITIONS MYAMG=Dune::Amg::AMG<Operator,Vector,Smoother>
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")
  add_dune_superlu_flags(pthreaddirectamgtest)
  add_dune_suitesparse_flags(pthreaddirectamgtest)

  dune_add_test(NAME pthreaddirectfastamgtest
                SOURCES pthreadamgtest.cc
                COMPILE_DEFINITIONS MYAMG=Dune::Amg::FastAMG<Operator,Vector>
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")
  add_dune_superlu_flags(pthreaddirectfastamgtest)
  add_dune_suitesparse_flags(pthreaddirectfastamgtest)

  dune_add_test(NAME pthreaddirecttwoleveltest
                SOURCES pthreadtwoleveltest.cc
                LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
                CMAKE_GUARD "SuperLU_FOUND OR SuiteSparse_UMFPACK_FOUND")
  add_dune_superlu_flags(pthreaddirecttwoleveltest)
  add_dune_suitesparse_flags(pthreaddirecttwoleveltest)
endif()

# add an executable without SuperLU/UMFPack
dune_add_test(NAME amgtest SOURCES amgtest.cc)

dune_add_test(NAME fastamg SOURCES fastamg.cc)

if(SuperLU_FOUND)
  dune_add_test(NAME superluamgtest SOURCES amgtest.cc)
  add_dune_superlu_flags(superluamgtest)

  dune_add_test(NAME superlufastamgtest SOURCES fastamg.cc)
  add_dune_superlu_flags(superlufastamgtest)
endif()

dune_add_test(NAME umfpackamgtest SOURCES amgtest.cc
              CMAKE_GUARD SuiteSparse_UMFPACK_FOUND)
add_dune_suitesparse_flags(umfpackamgtest)

dune_add_test(NAME umfpackfastamgtest SOURCES fastamg.cc
              CMAKE_GUARD SuiteSparse_UMFPACK_FOUND)
add_dune_suitesparse_flags(umfpackfastamgtest)

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
add_dune_mpi_flags(galerkintest)

dune_add_test(SOURCES hierarchytest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)
add_dune_mpi_flags(hierarchytest)
add_dune_parmetis_flags(hierarchytest) # ParMETIS is not a hard requirement

dune_add_test(SOURCES coarsentest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)
add_dune_mpi_flags(coarsentest)
add_dune_parmetis_flags(coarsentest) # ParMETIS is not a hard requirement

dune_add_test(NAME pamgtest
              SOURCES parallelamgtest.cc
              MPI_RANKS 1 2 4
              TIMEOUT 600
              CMAKE_GUARD MPI_FOUND)
add_dune_mpi_flags(pamgtest)
add_dune_parmetis_flags(pamgtest) # ParMETIS is not a hard requirement

dune_add_test(NAME pamg_comm_repart_test
              MPI_RANKS 1 2 4
              TIMEOUT 600
              SOURCES parallelamgtest.cc
              COMPILE_DEFINITIONS -DAMG_REPART_ON_COMM_GRAPH
              CMAKE_GUARD MPI_FOUND)
add_dune_mpi_flags(pamg_comm_repart_test)
add_dune_parmetis_flags(pamg_comm_repart_test) # ParMETIS is not a hard requirement

dune_add_test(NAME pamgmmtest
  SOURCES pamgmmtest.cc
  TIMEOUT 20
  MPI_RANKS 6
  CMAKE_GUARD "(HAVE_MPI AND HAVE_PARMETIS)")
add_dune_mpi_flags(pamgmmtest)
add_dune_parmetis_flags(pamgmmtest)

file(COPY gr_30_30.mm DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
