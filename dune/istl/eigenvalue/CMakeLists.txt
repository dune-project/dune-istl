add_subdirectory("test" EXCLUDE_FROM_ALL)

#install headers
install(FILES
   arpackpp.hh
   poweriteration.hh
   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/eigenvalue)
