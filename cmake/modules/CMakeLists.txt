set(modules
  AddSuperLUFlags.cmake
  AddUMFPackFlags.cmake
  DuneIstlMacros.cmake
  FindSuperLU.cmake
  FindUMFPack.cmake)

install(FILES ${modules} DESTINATION
  ${DUNE_INSTALL_MODULEDIR})
