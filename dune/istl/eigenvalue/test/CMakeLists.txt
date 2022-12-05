# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# Link all test targets in this directory against Dune::ISTL
link_libraries(Dune::ISTL)

dune_add_test(NAME poweriterationtest SOURCES cond2test.cc
  CMD_ARGS 40)

dune_add_test(NAME poweriterationsuperlutest SOURCES cond2test.cc
  CMD_ARGS 40
  CMAKE_GUARD "SuperLU_FOUND")
add_dune_superlu_flags(poweriterationsuperlutest)

dune_add_test(NAME arpackpptest SOURCES cond2test.cc
  CMAKE_GUARD "ARPACKPP_FOUND")
add_dune_arpackpp_flags(arpackpptest)

dune_add_test(NAME arpackppsuperlutest SOURCES cond2test.cc
  CMAKE_GUARD "(ARPACKPP_FOUND AND SuperLU_FOUND)")
add_dune_superlu_flags(arpackppsuperlutest)
add_dune_arpackpp_flags(arpackppsuperlutest)
