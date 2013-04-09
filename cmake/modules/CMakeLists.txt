set(modules DuneIstlMacros.cmake
  FindSuperLU.cmake)

install(FILES ${modules} DESTINATION
  ${DUNE_INSTALL_MODULEDIR})
