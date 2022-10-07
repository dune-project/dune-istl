// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SOLVERTYPE_HH
#define DUNE_ISTL_SOLVERTYPE_HH

/**
 * @file
 * @brief Templates characterizing the type of a solver.
 */
namespace Dune
{
  template<typename Solver>
  struct IsDirectSolver
  {
    enum
    {
      /**
       * @brief Whether this is a direct solver.
       *
       * If Solver is a direct solver, this is true.
       */
      value =false
    };
  };

  template<typename Solver>
  struct StoresColumnCompressed
  {
    enum
    {
      /**
       * @brief whether the solver internally uses column compressed storage
       */
      value = false
    };
  };
} // end namespace Dune
#endif
