add_subdirectory(test)

#install headers
install(FILES
   arpackpp.hh
   poweriteration.hh
   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/eigenvalue)
