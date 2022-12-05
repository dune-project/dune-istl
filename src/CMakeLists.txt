# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

dune_add_test(NAME istl-solver-playground
  SOURCES istl-solver-playground.cc LINK_LIBRARIES Dune::ISTL)
add_dune_mpi_flags(istl-solver-playground)
add_dune_parmetis_flags(istl-solver-playground)
dune_symlink_to_source_files(FILES playground.ini)
