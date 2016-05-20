// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SCHWARZ_HH
#define DUNE_ISTL_SCHWARZ_HH

#include <iostream>              // for input/output to shell
#include <fstream>               // for input/output to files
#include <vector>                // STL vector class
#include <sstream>

#include <cmath>                // Yes, we do some math here

#include <dune/common/timer.hh>

#include "io.hh"
#include "bvector.hh"
#include "vbvector.hh"
#include "bcrsmatrix.hh"
#include "io.hh"
#include "gsetc.hh"
#include "ilu.hh"
#include "operators.hh"
#include "solvers.hh"
#include "preconditioners.hh"
#include "scalarproducts.hh"
#include "owneroverlapcopy.hh"

namespace Dune {

  /**
   * @defgroup ISTL_Parallel Parallel Solvers
   * @ingroup ISTL_Solvers
   * Instead of using parallel data structures (matrices and vectors) that
   * (implicitly) know the data distribution and communication patterns,
   * there is a clear separation of the parallel data composition together
   *  with the communication APIs from the data structures. This allows for
   * implementing overlapping and nonoverlapping domain decompositions as
   * well as data parallel parallelisation approaches.
   *
   * The \ref ISTL_Solvers "solvers" can easily be turned into parallel solvers
   * initializing them with matching parallel subclasses of the base classes
   * ScalarProduct, Preconditioner and LinearOperator.
   *
   * The information of the data distribution is provided by OwnerOverlapCopyCommunication
   * of \ref ISTL_Comm "communication API".
   *
   * Currently only data parallel versions are shipped with dune-istl. Domain
   * decomposition can be found in module dune-dd.
   */
  /**
     @addtogroup ISTL_Operators
     @{
   */

  /**
   * \brief An overlapping schwarz operator.
   *
   * This operator represents a parallel matrix product using
   * sequential data structures together with a parallel index set
   * describing an overlapping domain decomposition and the communication.
   * \tparam M The type of the sequential matrix to use,
   * e.g. BCRSMatrix or another matrix type fulfilling the
   * matrix interface of ISTL.
   * \tparam X The type of the sequential vector to use for the left hand side,
   * e.g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam Y The type of the sequential vector to use for the right hand side,
   * e..g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam C The type of the communication object.
   * This must either be OwnerOverlapCopyCommunication or a type
   * implementing the same interface.
   */
  template<class M, class X, class Y, class C>
  class OverlappingSchwarzOperator : public AssembledLinearOperator<M,X,Y>
  {
  public:
    //! \brief The type of the matrix we operate on.
    //!
    //! E.g. BCRSMatrix or another matrix type fulfilling the
    //! matrix interface of ISTL
    typedef M matrix_type;
    //! \brief The type of the domain.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef X domain_type;
    //! \brief The type of the range.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef Y range_type;
    //! \brief The field type of the range
    typedef typename X::field_type field_type;
    //! \brief The type of the communication object.
    //!
    //! This must either be OwnerOverlapCopyCommunication or a type
    //! implementing the same interface.
    typedef C communication_type;

    enum {
      //! \brief The solver category.
      category=SolverCategory::overlapping
    };

    /**
     * @brief constructor: just store a reference to a matrix.
     *
     * @param A The assembled matrix.
     * @param com The communication object for syncing overlap and copy
     * data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    OverlappingSchwarzOperator (const matrix_type& A, const communication_type& com)
      : _A_(A), communication(com)
    {}

    //! apply operator to x:  \f$ y = A(x) \f$
    virtual void apply (const X& x, Y& y) const
    {
      y = 0;
      _A_.umv(x,y);     // result is consistent on interior+border
      communication.project(y);     // we want this here to avoid it before the preconditioner
                                    // since there d is const!
    }

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    virtual void applyscaleadd (field_type alpha, const X& x, Y& y) const
    {
      _A_.usmv(alpha,x,y);     // result is consistent on interior+border
      communication.project(y);     // we want this here to avoid it before the preconditioner
                                    // since there d is const!
    }

    //! get the sequential assembled linear operator.
    virtual const matrix_type& getmat () const
    {
      return _A_;
    }

  private:
    const matrix_type& _A_;
    const communication_type& communication;
  };

  /** @} */

  /**
   * @addtogroup ISTL_SP
   * @{
   */
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
    typedef typename FieldTraits<field_type>::real_type real_type;
    //! \brief The type of the communication object.
    //!
    //! This must either be OwnerOverlapCopyCommunication or a type
    //! implementing the same interface.
    typedef C communication_type;

    //! define the category
    enum {category=SolverCategory::overlapping};

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

  template<class X, class C>
  struct ScalarProductChooser<X,C,SolverCategory::overlapping>
  {
    /** @brief The type of the scalar product for the overlapping case. */
    typedef OverlappingSchwarzScalarProduct<X,C> ScalarProduct;
    /** @brief The type of the communication object to use. */
    typedef C communication_type;

    enum {
      /** @brief The solver category. */
      solverCategory=SolverCategory::overlapping
    };

    static ScalarProduct* construct(const communication_type& comm)
    {
      return new ScalarProduct(comm);
    }
  };

  /**
   * @}
   *
   * @addtogroup ISTL_Prec
   * @{
   */
  //! \brief A parallel SSOR preconditioner.
  //! \tparam M The type of the sequential matrix to use,
  //! e.g. BCRSMatrix or another matrix type fulfilling the
  //! matrix interface of ISTL.
  //! \tparam X The type of the sequential vector to use for the left hand side,
  //! e.g. BlockVector or another type fulfilling the ISTL
  //! vector interface.
  //! \tparam Y The type of the sequential vector to use for the right hand side,
  //! e..g. BlockVector or another type fulfilling the ISTL
  //! vector interface.
  //! \tparam C The type of the communication object.
  //! This must either be OwnerOverlapCopyCommunication or a type
  //! implementing the same interface.
  template<class M, class X, class Y, class C>
  class ParSSOR : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief The type of the communication object.
    typedef C communication_type;

    // define the category
    enum {
      //! \brief The category the precondtioner is part of.
      category=SolverCategory::overlapping
    };

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param n The number of iterations to perform.
       \param w The relaxation factor.
       \param c The communication object for syncing overlap and copy
     * data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    ParSSOR (const matrix_type& A, int n, field_type w, const communication_type& c)
      : _A_(A), _n(n), _w(w), communication(c)
    {   }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      communication.copyOwnerToAll(x,x);     // make dirichlet values consistent
    }

    /*!
       \brief Apply the precondtioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      for (int i=0; i<_n; i++) {
        bsorf(_A_,v,d,_w);
        bsorb(_A_,v,d,_w);
      }
      communication.copyOwnerToAll(v,v);
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {}

  private:
    //! \brief The matrix we operate on.
    const matrix_type& _A_;
    //! \brief The number of steps to do in apply
    int _n;
    //! \brief The relaxation factor to use
    field_type _w;
    //! \brief the communication object
    const communication_type& communication;
  };

  namespace Amg
  {
    template<class T> class ConstructionTraits;
  }

  /**
   * @brief Block parallel preconditioner.
   *
   * This is essentially a wrapper that takes a sequential
   * preconditioner. In each step the sequential preconditioner
   * is applied and then all owner data points are updated on
   * all other processes.
   * \tparam M The type of the sequential matrix to use,
   * e.g. BCRSMatrix or another matrix type fulfilling the
   * matrix interface of ISTL.
   * \tparam X The type of the sequential vector to use for the left hand side,
   * e.g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam Y The type of the sequential vector to use for the right hand side,
   * e..g. BlockVector or another type fulfilling the ISTL
   * vector interface.
   * \tparam C The type of the communication object.
   * This must either be OwnerOverlapCopyCommunication or a type
   * implementing the same interface.
   * \tparam The type of the sequential preconditioner to use
   * for approximately solving the local matrix block consisting of unknowns
   * owned by the process. Has to implement the Preconditioner interface.
   */
  template<class X, class Y, class C, class T=Preconditioner<X,Y> >
  class BlockPreconditioner : public Preconditioner<X,Y> {
    friend class Amg::ConstructionTraits<BlockPreconditioner<X,Y,C,T> >;
  public:
    //! \brief The domain type of the preconditioner.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    //!
    //! E.g. BlockVector or another type fulfilling the ISTL
    //! vector interface.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief The type of the communication object..
    //!
    //! This must either be OwnerOverlapCopyCommunication or a type
    //! implementing the same interface.
    typedef C communication_type;

    // define the category
    enum {
      //! \brief The category the precondtioner is part of.
      category=SolverCategory::overlapping
    };

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param p The sequential preconditioner.
       \param c The communication object for syncing overlap and copy
       data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    BlockPreconditioner (T& p, const communication_type& c)
      : preconditioner(p), communication(c)
    {   }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b)
    {
      communication.copyOwnerToAll(x,x);     // make dirichlet values consistent
      preconditioner.pre(x,b);
    }

    /*!
       \brief Apply the preconditioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      preconditioner.apply(v,d);
      communication.copyOwnerToAll(v,v);
    }

    template<bool forward>
    void apply (X& v, const Y& d)
    {
      preconditioner.template apply<forward>(v,d);
      communication.copyOwnerToAll(v,v);
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x)
    {
      preconditioner.post(x);
    }

  private:
    //! \brief a sequential preconditioner
    T& preconditioner;

    //! \brief the communication object
    const communication_type& communication;
  };

  /** @} end documentation */

} // end namespace

#endif
