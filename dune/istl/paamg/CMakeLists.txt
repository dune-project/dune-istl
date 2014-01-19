add_subdirectory("test" EXCLUDE_FROM_ALL)

#install headers
install(FILES
  aggregates.hh
  amg.hh
  combinedfunctor.hh
  construction.hh
  dependency.hh
  fastamg.hh
  fastamgsmoother.hh
  galerkin.hh
  globalaggregates.hh
  graph.hh
  graphcreator.hh
  hierarchy.hh
  indicescoarsener.hh
  kamg.hh
  parameters.hh
  pinfo.hh
  properties.hh
  renumberer.hh
  smoother.hh
  transfer.hh
  twolevelmethod.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/paamg)
