set(MPITESTS  galerkintest hierarchytest pamgtest
  pamg_comm_repart_test transfertest)
set(NORMALTESTS amgtest graphtest kamgtest)
set(ALLTESTS ${MPITESTS} ${NORMALTESTS})

# We do not want want to build the tests during make all,
# but just build them on demand
add_dependencies(build_tests ${ALLTESTS})

add_executable(amgtest "amgtest.cc")
add_executable(galerkintest "galerkintest.cc")
add_executable(graphtest "graphtest.cc")
add_executable(hierarchytest "hierarchytest.cc")
add_executable(kamgtest "kamgtest.cc")
add_executable(pamg_comm_repart_test "parallelamgtest.cc")
set_target_properties(pamg_comm_repart_test PROPERTIES COMPILE_FLAGS "-DAMG_REPART_ON_COMM_GRAPH")
add_executable(pamgtest "parallelamgtest.cc")
add_executable(transfertest "transfertest.cc")

message("MPITESTS=${MPITESTS}")

message("ALLTESTS=${ALLTESTS}")

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
  add_test(${_exe} ${_exe})
endforeach(_exe ${ALLTESTS})

include(DuneMPI)
add_dune_mpi_flags("${MPITESTS}")
