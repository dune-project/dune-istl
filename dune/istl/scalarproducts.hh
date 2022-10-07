// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SCALARPRODUCTS_HH
#define DUNE_ISTL_SCALARPRODUCTS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>
#include <memory>

#include <dune/common/exceptions.hh>
#include <dune/common/shared_ptr.hh>

#include "bvector.hh"
#include "solvercategory.hh"


namespace Dune {
  /**
   * @defgroup ISTL_SP Scalar products
   * @ingroup ISTL_Solvers
   * @brief Scalar products for the use in iterative solvers
   */
  /** @addtogroup ISTL_SP
          @{
   */

  /** \file

     \brief Define base class for scalar product and norm.

     These classes have to be implemented differently for different
     data partitioning strategies. Default implementations for the
     sequential case are provided.

   */

  /*! \brief Base class for scalar product and norm computation

      Krylov space methods need to compute scalar products and norms
      (for convergence test only). These methods have to know about the
          underlying data decomposition. For the sequential case a default implementation
          is provided.

      by default the scalar product is sequential
   */
  template<class X>
  class ScalarProduct {
  public:
    //! export types, they come from the derived class
    typedef X domain_type;
    typedef typename X::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;

    /*! \brief Dot product of two vectors.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y) const
    {
      return x.dot(y);
    }

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x) const
    {
      return x.two_norm();
    }

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

    //! every abstract base class has a virtual destructor
    virtual ~ScalarProduct () {}
  };

  /**
   * \brief Scalar product for overlapping Schwarz methods.
   *
   * Consistent vectors in interior and border are assumed.
   * \tparam  X The type of the sequential vector to use for the left hand side,
   * e.g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam C The type of the communication object.
   * This must either be OwnerOverlapCopyCommunication or a type
   * implementing the same interface.
   */
  template<class X, class C>
  class ParallelScalarProduct : public ScalarProduct<X>
  {
  public:
    //! \brief The type of the vector to compute the scalar product on.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef X domain_type;
    //!  \brief The field type used by the vector type domain_type.
    typedef typename X::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;
    //! \brief The type of the communication object.
    //!
    //! This must either be OwnerOverlapCopyCommunication or a type
    //! implementing the same interface.
    typedef C communication_type;

    /*!
     * \param com The communication object for syncing overlap and copy
     * data points.
     * \param cat parallel solver category (nonoverlapping or overlapping)
     */
    ParallelScalarProduct (std::shared_ptr<const communication_type> com, SolverCategory::Category cat)
      : _communication(com), _category(cat)
    {}

    /*!
     * \param com The communication object for syncing overlap and copy
     * data points.
     * \param cat parallel solver category (nonoverlapping or overlapping)
     * \note if you use this constructor you have to make sure com stays alive
     */
    ParallelScalarProduct (const communication_type& com, SolverCategory::Category cat)
      : ParallelScalarProduct(stackobject_to_shared_ptr(com), cat)
    {}


    /*! \brief Dot product of two vectors.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y) const override
    {
      field_type result(0);
      _communication->dot(x,y,result); // explicitly loop and apply masking
      return result;
    }

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x) const override
    {
      return _communication->norm(x);
    }

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const override
    {
      return _category;
    }

  private:
    std::shared_ptr<const communication_type> _communication;
    SolverCategory::Category _category;
  };

  //! Default implementation for the scalar case
  template<class X>
  class SeqScalarProduct : public ScalarProduct<X>
  {
    using ScalarProduct<X>::ScalarProduct;
  };

  /**
   * \brief Nonoverlapping Scalar Product with communication object.
   *
   * Consistent vectors in interior and border are assumed.
   */
  template<class X, class C>
  class NonoverlappingSchwarzScalarProduct : public ParallelScalarProduct<X,C>
  {
  public:
    NonoverlappingSchwarzScalarProduct (std::shared_ptr<const C> comm) :
      ParallelScalarProduct<X,C>(comm,SolverCategory::nonoverlapping) {}

    NonoverlappingSchwarzScalarProduct (const C& comm) :
      ParallelScalarProduct<X,C>(comm,SolverCategory::nonoverlapping) {}
  };

  /**
   * \brief Scalar product for overlapping Schwarz methods.
   *
   * Consistent vectors in interior and border are assumed.
   * \tparam  X The type of the sequential vector to use for the left hand side,
   * e.g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam C The type of the communication object.
   * This must either be OwnerOverlapCopyCommunication or a type
   * implementing the same interface.
   */
  template<class X, class C>
  class OverlappingSchwarzScalarProduct : public ParallelScalarProduct<X,C>
  {
  public:
    OverlappingSchwarzScalarProduct (std::shared_ptr<const C> comm) :
      ParallelScalarProduct<X,C>(comm, SolverCategory::overlapping) {}

    OverlappingSchwarzScalarProduct (const C& comm) :
      ParallelScalarProduct<X,C>(comm,SolverCategory::overlapping) {}
  };

  /** @} end documentation */

  /**
   * \brief Choose the approriate scalar product for a solver category.
   *
   * \todo this helper function should be replaced by a proper factory
   *
   * As there is only one scalar product for each solver category it is
   * possible to choose the appropriate product at compile time.
   *
   * In each specialization of the this struct there will be a typedef ScalarProduct
   * available the defines the type  of the scalar product.
   */
  template<class X, class Comm>
  std::shared_ptr<ScalarProduct<X>> makeScalarProduct(std::shared_ptr<const Comm> comm, SolverCategory::Category category)
  {
    switch(category)
    {
      case SolverCategory::sequential:
        return
          std::make_shared<ScalarProduct<X>>();
      default:
        return
          std::make_shared<ParallelScalarProduct<X,Comm>>(comm,category);
    }
  }

  /**
   * \copydoc createScalarProduct(std::shared_ptr<const Comm>,SolverCategory::Category)
   * \note Using this helper, you are responsible for the life-time management of comm
   */
  template<class X, class Comm>
  std::shared_ptr<ScalarProduct<X>> createScalarProduct(const Comm& comm, SolverCategory::Category category)
  { return makeScalarProduct<X>(stackobject_to_shared_ptr(comm), category); }

} // end namespace Dune

#endif
