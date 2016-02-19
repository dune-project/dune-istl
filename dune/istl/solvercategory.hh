// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SOLVERCATEGORY_HH
#define DUNE_ISTL_SOLVERCATEGORY_HH


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
  };

  /** @} end documentation */

} // end namespace

#endif
