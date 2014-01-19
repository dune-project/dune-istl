set(modules
  AddSuperLUFlags.cmake
  DuneIstlMacros.cmake
  FindSuperLU.cmake)

install(FILES ${modules} DESTINATION
  ${DUNE_INSTALL_MODULEDIR})
