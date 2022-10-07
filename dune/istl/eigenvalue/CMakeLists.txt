# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

add_subdirectory(test)

#install headers
install(FILES
   arpackpp.hh
   poweriteration.hh
   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl/eigenvalue)
