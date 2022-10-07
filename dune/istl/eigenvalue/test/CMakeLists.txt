# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

dune_add_test(NAME poweriterationtest SOURCES cond2test.cc
  CMD_ARGS 40)

if(SuperLU_FOUND)
  dune_add_test(NAME poweriterationsuperlutest SOURCES cond2test.cc
    CMD_ARGS 40)
  add_dune_superlu_flags(poweriterationsuperlutest)
endif()

if(ARPACKPP_FOUND)
  dune_add_test(NAME arpackpptest SOURCES cond2test.cc)
  add_dune_arpackpp_flags(arpackpptest)
endif()

if(ARPACKPP_FOUND AND SuperLU_FOUND)
  dune_add_test(NAME arpackppsuperlutest SOURCES cond2test.cc)
endif()
