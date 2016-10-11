// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SCALARPRODUCTS_HH
#define DUNE_ISTL_SCALARPRODUCTS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>

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
   */
  template<class X>
  class ScalarProduct {
  public:

    //! export types, they come from the derived class
    typedef X domain_type;
    typedef typename X::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const = 0;

    /*! \brief Dot product of two vectors.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y) = 0;

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x) = 0;

    //! every abstract base class has a virtual destructor
    virtual ~ScalarProduct () {}
  };

  //=====================================================================
  // Implementation for ISTL-matrix based operator
  //=====================================================================

  //! Default implementation for the scalar case
  template<class X>
  class SeqScalarProduct : public ScalarProduct<X>
  {
  public:

    //! export types
    typedef X domain_type;
    typedef typename X::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

    /*! \brief Dot product of two vectors. In the complex case, the first argument is conjugated.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y)
    {
      return x.dot(y);
    }

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x)
    {
      return x.two_norm();
    }
  };


  /**
   * \brief Nonoverlapping Scalar Product with communication object.
   *
   * Consistent vectors in interior and border are assumed.
   */
  template<class X, class C>
  class NonoverlappingSchwarzScalarProduct : public ScalarProduct<X>
  {
  public:
    //! \brief The type of the domain.
    typedef X domain_type;
    //!  \brief The type of the range
    typedef typename X::field_type field_type;
    //!  \brief The real-type of the range
    typedef typename FieldTraits<field_type>::real_type real_type;
    //! \brief The type of the communication object
    typedef C communication_type;

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::nonoverlapping;
    }

    /*! \brief Constructor
     * \param com The communication object for syncing owner and copy
     * data points. (E.~g. OwnerOverlapCommunication )
     */
    NonoverlappingSchwarzScalarProduct (const communication_type& com)
      : communication(com)
    {}

    /*! \brief Dot product of two vectors.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y)
    {
      field_type result;
      communication.dot(x,y,result);
      return result;
    }

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x)
    {
      return communication.norm(x);
    }

    /*! \brief make additive vector consistent
     */
    void make_consistent (X& x) const
    {
      communication.copyOwnerToAll(x,x);
    }

  private:
    const communication_type& communication;
  };


  /**
   * \brief Scalar product for overlapping schwarz methods.
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
  class OverlappingSchwarzScalarProduct : public ScalarProduct<X>
  {
  public:
    //! \brief The type of the vector to compute the scalar product on.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef X domain_type;
    //!  \brief The field type used by the vector type domain_type.
    typedef typename X::field_type field_type;
    //!  \brief The real-type of the range
    typedef typename FieldTraits<field_type>::real_type real_type;
    //! \brief The type of the communication object.
    //!
    //! This must either be OwnerOverlapCopyCommunication or a type
    //! implementing the same interface.
    typedef C communication_type;

    //! Category of the scalar product (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::overlapping;
    }

    /*! \brief Constructor needs to know the grid
     * \param com The communication object for syncing overlap and copy
     * data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    OverlappingSchwarzScalarProduct (const communication_type& com)
      : communication(com)
    {}

    /*! \brief Dot product of two vectors.
       It is assumed that the vectors are consistent on the interior+border
       partition.
     */
    virtual field_type dot (const X& x, const X& y)
    {
      field_type result;
      communication.dot(x,y,result);
      return result;
    }

    /*! \brief Norm of a right-hand side vector.
       The vector must be consistent on the interior+border partition
     */
    virtual real_type norm (const X& x)
    {
      return communication.norm(x);
    }

  private:
    const communication_type& communication;
  };


  /*
   * \brief Choose the approriate scalar product for a solver category.
   */
  class ScalarProductChooser
  {
  public:
    template <class X, class COMM>
    static std::shared_ptr<ScalarProduct<X> > construct (const SolverCategory::Category category, const COMM& comm) {
      if (category == SolverCategory::sequential)
        return std::make_shared<SeqScalarProduct<X> > ();
      if (category == SolverCategory::nonoverlapping)
        return std::make_shared<NonoverlappingSchwarzScalarProduct<X,COMM> > (comm);
      if (category == SolverCategory::overlapping)
        return std::make_shared<OverlappingSchwarzScalarProduct<X,COMM> > (comm);
      DUNE_THROW(ISTLError,"ScalarProductChooser called with unknown category!");
    }
  };

  /** @} end documentation */

} // end namespace

#endif
