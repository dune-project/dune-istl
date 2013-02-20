add_subdirectory("test" EXCLUDE_FROM_ALL)

#install headers
install(FILES aggregates.hh dependency.hh galerkin.hh graph.hh
          indicescoarsener.hh properties.hh globalaggregates.hh
          hierarchy.hh construction.hh
          transfer.hh smoother.hh amg.hh kamg.hh combinedfunctor.hh
          graphcreator.hh parameters.hh renumberer.hh pinfo.hh
	DESTINATION include/dune/istl/paamg)
