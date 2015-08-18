set(modules
  AddARPACKPPFlags.cmake
  AddSuperLUFlags.cmake
  DuneIstlMacros.cmake
  FindARPACK.cmake
  FindARPACKPP.cmake
  FindSuperLU.cmake)

install(FILES ${modules} DESTINATION
  ${DUNE_INSTALL_MODULEDIR})
