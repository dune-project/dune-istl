# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

# .. cmake_module::
#
#    This modules content is executed whenever a module required or suggests dune-istl!
#

find_package(METIS)
find_package(ParMETIS)
include(AddParMETISFlags)
find_package(SuperLU 5.0)
include(AddSuperLUFlags)
find_package(ARPACKPP)
include(AddARPACKPPFlags)
find_package(SuiteSparse OPTIONAL_COMPONENTS CHOLMOD LDL SPQR UMFPACK)
include(AddSuiteSparseFlags)

# enable / disable backwards compatibility w.r.t. category
set(DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE 1
  CACHE BOOL "Enable/Disable the backwards compatibility of the category enum/method in dune-istl solvers, preconditioner, etc. '1'")
