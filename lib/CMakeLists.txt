install(FILES duneistlam2cmake.lib
  DESTINATION ${CMAKE_INSTALL_BINDIR}/../lib)

dune_add_library("duneistl" SHARED
  solverrepository.cc)
add_dune_all_flags(duneistl)
