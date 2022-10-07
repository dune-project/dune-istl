// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SOLVERCATEGORY_HH
#define DUNE_ISTL_SOLVERCATEGORY_HH

#include <dune/common/exceptions.hh>


namespace Dune {

  /**
     @addtogroup ISTL_Solvers
     @{
   */

  /**
   * @brief Categories for the solvers.
   */
  struct SolverCategory
  {
    enum  Category {
      //! \brief Category for sequential solvers
      sequential,
      //! \brief Category for non-overlapping solvers
      nonoverlapping,
      //! \brief Category for overlapping solvers
      overlapping
    };

    /**  \brief Helperfunction to extract the solver category either from an enum, or from the newly introduced virtual member function */
    template<typename OP>
    static Category category(const OP& op, decltype(op.category())* = nullptr)
    {
      return op.category();
    }

#ifndef DOXYGEN
    // template<typename OP>
    // static Category category(const OP& op, decltype(op.getSolverCategory())* = nullptr)
    // {
    //   return op.getSolverCategory();
    // }

    template<typename OP>
    static Category category(const OP& op, decltype(op.category)* = nullptr)
    {
      return OP::category;
    }
#endif
  };

  class InvalidSolverCategory : public InvalidStateException{};

  /** @} end documentation */

} // end namespace

#endif
