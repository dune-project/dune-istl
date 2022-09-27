# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

add_executable(istl-solver-playground EXCLUDE_FROM_ALL istl-solver-playground.cc)
add_dune_all_flags(istl-solver-playground)
dune_symlink_to_source_files(FILES playground.ini)
dune_add_test(TARGET istl-solver-playground)
