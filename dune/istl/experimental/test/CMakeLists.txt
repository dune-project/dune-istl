# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Link all test targets in this directory against Dune::ISTL
link_libraries(Dune::ISTL)

dune_add_test(NAME experimentalsparseindexrangestest
              SOURCES sparseindexrangestest.cc)

dune_add_test(NAME experimentalmatrixviewtest
              SOURCES matrixviewtest.cc
              COMPILE_DEFINITIONS DUNE_ISTL_WITH_CHECKING=1)

dune_add_test(NAME experimentalmatrixtest
              SOURCES matrixtest.cc)

dune_add_test(NAME experimentalbcrsmatrixtest
              SOURCES bcrsmatrixtest.cc)

dune_add_test(NAME experimentalindexedarrayviewtest
              SOURCES indexedarrayviewtest.cc)

dune_add_test(NAME experimentalvectorviewtest
              SOURCES vectorviewtest.cc
              COMPILE_DEFINITIONS DUNE_ISTL_WITH_CHECKING=1)
