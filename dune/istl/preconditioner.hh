// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PRECONDITIONER_HH
#define DUNE_ISTL_PRECONDITIONER_HH

#include <dune/common/exceptions.hh>

#include "solvercategory.hh"

namespace Dune {
/**
 * @addtogroup ISTL_Prec
 * @{
 */
  //=====================================================================
  /*! \brief Base class for matrix free definition of preconditioners.

     Note that the operator, which is the basis for the preconditioning,
     is supplied to the preconditioner from the outside in the
     constructor or some other method.

     This interface allows the encapsulation of all parallelization
     aspects into the preconditioners.

     \tparam X Type of the update
     \tparam Y Type of the defect
    */
  //=====================================================================
  template<class X, class Y>
  class Preconditioner {
  public:
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;

    /*! \brief Prepare the preconditioner.

       A solver solves a linear operator equation A(x)=b by applying
       one or several steps of the preconditioner. The method pre()
       is called before the first apply operation.
       b and x are right hand side and solution vector of the linear
       system respectively. It may. e.g., scale the system, allocate memory or
       compute a (I)LU decomposition.
       Note: The ILU decomposition could also be computed in the constructor
       or with a separate method of the derived method if several
       linear systems with the same matrix are to be solved.

       \note if a preconditioner is copied (e.g. for a second thread)
       again the pre() method has to be called to ensure proper memory
       mangement.

       \code
       X x(0.0);
       Y b = ...; // rhs
       Preconditioner<X,Y> prec(...);
       prec.pre(x,b);   // prepare the preconditioner
       prec.apply(x,b); // can be called multiple times now...
       prec.post(x);    // cleanup internal state
       \endcode

       \param x The left hand side of the equation.
       \param b The right hand side of the equation.
     */
    virtual void pre (X& x, Y& b) = 0;

    /*! \brief Apply one step of the preconditioner to the system A(v)=d.

       On entry v=0 and d=b-A(x) (although this might not be
       computed in that way. On exit v contains the update, i.e
       one step computes \f$ v = M^{-1} d \f$ where \f$ M \f$ is the
       approximate inverse of the operator \f$ A \f$ characterizing
       the preconditioner.
       \param[out] v The update to be computed
       \param d The current defect.
     */
    virtual void apply (X& v, const Y& d) = 0;

    /*! \brief Clean up.

       This method is called after the last apply call for the
       linear system to be solved. Memory may be deallocated safely
       here. x is the solution of the linear equation.

       \param x The right hand side of the equation.
     */
    virtual void post (X& x) = 0;

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
#if DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE
    {
      DUNE_THROW(Dune::Exception,"It is necessary to implement the category method in a derived classes, in the future this method will pure virtual.");
    }
#else
    = 0;
#endif

    //! every abstract base class has a virtual destructor
    virtual ~Preconditioner () {}

  };

/**
 * @}
 */
}
#endif
