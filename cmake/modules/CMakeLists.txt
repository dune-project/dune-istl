# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

set(modules
  AddARPACKPPFlags.cmake
  AddSuperLUFlags.cmake
  DuneIstlMacros.cmake
  FindARPACK.cmake
  FindARPACKPP.cmake
  FindSuperLU.cmake)

install(FILES ${modules} DESTINATION
  ${DUNE_INSTALL_MODULEDIR})
