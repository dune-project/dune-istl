// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_OPERATORS_HH
#define DUNE_ISTL_OPERATORS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>

#include <dune/common/exceptions.hh>
#include <dune/common/shared_ptr.hh>

#include "solvercategory.hh"


namespace Dune {

  /**
   * @defgroup ISTL_Operators Operator concept
   * @ingroup ISTL_Solvers
   *
   * The solvers in ISTL do not work on matrices directly. Instead we use
   * an abstract operator concept. This allows for using matrix-free
   * operators, i.e. operators that are not stored as matrices in any
   * form. Thus our solver algorithms can easily be turned into matrix-free
   * solvers just by plugging in matrix-free representations of linear
   * operators and preconditioners.
   */
  /** @addtogroup ISTL_Operators
          @{
   */


  /** \file

     \brief Define general, extensible interface for operators.
          The available implementation wraps a matrix.
   */

  //=====================================================================
  // Abstract operator interface
  //=====================================================================


  /*!
     @brief A linear operator.

     Abstract base class defining a linear operator \f$ A : X\to Y\f$,
     i.e. \f$ A(\alpha x) = \alpha A(x) \f$ and
      \f$ A(x+y) = A(x)+A(y)\f$ hold. The
     simplest solvers just need the application  \f$ A(x)\f$ of
     the operator.


        - enables on the fly computation through operator concept. If explicit
        representation of the operator is required use AssembledLinearOperator

        - Some inverters may need an explicit formation of the operator
        as a matrix, e.g. BiCGStab, ILU, AMG, etc. In that case use the
        derived class
   */
  template<class X, class Y>
  class LinearOperator {
  public:
    //! The type of the domain of the operator.
    typedef X domain_type;
    //! The type of the range of the operator.
    typedef Y range_type;
    //! The field type of the operator.
    typedef typename X::field_type field_type;

    /*! \brief apply operator to x:  \f$ y = A(x) \f$
          The input vector is consistent and the output must also be
       consistent on the interior+border partition.
     */
    virtual void apply (const X& x, Y& y) const = 0;

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    virtual void applyscaleadd (field_type alpha, const X& x, Y& y) const = 0;

    //! every abstract base class has a virtual destructor
    virtual ~LinearOperator () {}

    //! Category of the linear operator (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
#if DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE
    {
      DUNE_THROW(Dune::Exception,"It is necessary to implement the category method in a derived classes, in the future this method will pure virtual.");
    };
#else
    = 0;
#endif
  };


  /*!
     \brief A linear operator exporting itself in matrix form.

     Linear Operator that exports the operator in
     matrix form. This is needed for certain solvers, such as
     LU decomposition, ILU preconditioners or BiCG-Stab (because
     of multiplication with A^T).
   */
  template<class M, class X, class Y>
  class AssembledLinearOperator : public LinearOperator<X,Y> {
  public:
    //! export types, usually they come from the derived class
    typedef M matrix_type;
    typedef X domain_type;
    typedef Y range_type;
    typedef typename X::field_type field_type;

    //! get matrix via *
    virtual const M& getmat () const = 0;

    //! every abstract base class has a virtual destructor
    virtual ~AssembledLinearOperator () {}
  };



  //=====================================================================
  // Implementation for ISTL-matrix based operator
  //=====================================================================

  /*!
     \brief Adapter to turn a matrix into a linear operator.

     Adapts a matrix to the assembled linear operator interface
   */
  template<class M, class X, class Y>
  class MatrixAdapter : public AssembledLinearOperator<M,X,Y>
  {
  public:
    //! export types
    typedef M matrix_type;
    typedef X domain_type;
    typedef Y range_type;
    typedef typename X::field_type field_type;

    //! constructor: just store a reference to a matrix
    explicit MatrixAdapter (const M& A) : _A_(stackobject_to_shared_ptr(A)) {}

    //! constructor: store an std::shared_ptr to a matrix
    explicit MatrixAdapter (std::shared_ptr<const M> A) : _A_(A) {}

    //! apply operator to x:  \f$ y = A(x) \f$
    void apply (const X& x, Y& y) const override
    {
      _A_->mv(x,y);
    }

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    void applyscaleadd (field_type alpha, const X& x, Y& y) const override
    {
      _A_->usmv(alpha,x,y);
    }

    //! get matrix via *
    const M& getmat () const override
    {
      return *_A_;
    }

    //! Category of the solver (see SolverCategory::Category)
    SolverCategory::Category category() const override
    {
      return SolverCategory::sequential;
    }

  private:
    const std::shared_ptr<const M> _A_;
  };

  /** @} end documentation */

} // end namespace

#endif
