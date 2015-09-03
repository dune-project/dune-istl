add_executable(example "example.cc")
add_dune_mpi_flags(example)
target_link_libraries(example "dunecommon")
