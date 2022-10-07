# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

set(HEADERS
  bcrsmatrix.hh
  bvector.hh
  iterator.hh
  matrixindexset.hh
  operators.hh
  preconditioners.hh
  slice.hh
  solvers.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/python/istl)
