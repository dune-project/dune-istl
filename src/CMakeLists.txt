# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

dune_add_test(NAME istl-solver-playground
  SOURCES istl-solver-playground.cc)
dune_symlink_to_source_files(FILES playground.ini)
