set(MPITESTS  galerkintest hierarchytest pamgtest
  pamg_comm_repart_test transfertest)

set(NORMALTEST amgtest graphtest kamgtest)

# We do not want want to build the tests during make all,
# but just build them on demand
add_dependencies(build_tests ${MPITESTS} ${NORMALTEST})

add_executable(amgtest "amgtest.cc")
add_executable(galerkintest "galerkintest.cc")
add_executable(graphtest "graphtest.cc")
add_executable(hierarchytest "hierarchytest.cc")
add_executable(kamgtest "kamgtest.cc")
add_executable(pamg_comm_repart_test "parallelamgtest.cc")
set_target_properties(pamg_comm_repart_test PROPERTIES COMPILE_FLAGS "-DAMG_REPART_ON_COMM_GRAPH")
add_executable(pamgtest "parallelamgtest.cc")
add_executable(transfertest "transfertest.cc")

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
endforeach(_exe ${ALLTESTS})

include(DuneMPI)
add_dune_mpi_flags(${MPITESTS})
