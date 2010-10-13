// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_NOVLPSCHWARZ_HH
#define DUNE_NOVLPSCHWARZ_HH

#include <iostream>              // for input/output to shell
#include <fstream>               // for input/output to files
#include <vector>                // STL vector class
#include <sstream>

#include <cmath>                // Yes, we do some math here
#include <sys/times.h>           // for timing measurements

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
   * initializing them with matching parallel subclasses of the the base classes
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
   * @brief A nonoverlapping operator with communication object.
   */
  template<class M, class X, class Y, class C>
  class NonoverlappingSchwarzOperator : public AssembledLinearOperator<M,X,Y>
  {
  public:
    //! \brief The type of the matrix we operate on.
    typedef M matrix_type;
    //! \brief The type of the domain.
    typedef X domain_type;
    //! \brief The type of the range.
    typedef Y range_type;
    //! \brief The field type of the range
    typedef typename X::field_type field_type;
    //! \brief The type of the communication object
    typedef C communication_type;

    enum {
      //! \brief The solver category.
      category=SolverCategory::nonoverlapping
    };

    /**
     * @brief constructor: just store a reference to a matrix.
     *
     * @param A The assembled matrix.
     * @param com The communication object for syncing owner and copy
     * data points. (E.~g. OwnerOverlapCommunication )
     */
    NonoverlappingSchwarzOperator (const matrix_type& A, const communication_type& com)
      : _A_(A), communication(com)
    {}

    //! apply operator to x:  \f$ y = A(x) \f$
    virtual void apply (const X& x, Y& y) const
    {
      y = 0;
      communication.novlp_op_apply(_A_,x,y,1);
      communication.addOwnerCopyToAll(y,y);
    }

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    virtual void applyscaleadd (field_type alpha, const X& x, Y& y) const
    {
      communication.novlp_op_apply(_A_,x,y,alpha);
      communication.addOwnerCopyToAll(y,y);
    }

    //! get matrix via *
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
    //! \brief The type of the communication object
    typedef C communication_type;

    //! define the category
    enum {category=SolverCategory::nonoverlapping};

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
    virtual double norm (const X& x)
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

  template<class X, class C>
  struct ScalarProductChooser<X,C,SolverCategory::nonoverlapping>
  {
    /** @brief The type of the scalar product for the nonoverlapping case. */
    typedef NonoverlappingSchwarzScalarProduct<X,C> ScalarProduct;
    /** @brief The type of the communication object to use. */
    typedef C communication_type;

    enum {
      /** @brief The solver category. */
      solverCategory=SolverCategory::nonoverlapping
    };

    static ScalarProduct* construct(const communication_type& comm)
    {
      return new ScalarProduct(comm);
    }
  };

  namespace Amg
  {
    template<class T> class ConstructionTraits;
  }

  /**
   * @brief Nonoverlapping parallel preconditioner.
   *
   * This is essentially a wrapper that take a sequential
   * preconditoner. In each step the sequential preconditioner
   * is applied and then all owner data points are updated on
   * all other processes.
   */

  template<class C, class P>
  class NonoverlappingBlockPreconditioner
    : public Dune::Preconditioner<typename P::domain_type,typename P::range_type> {
    friend class Amg::ConstructionTraits<NonoverlappingBlockPreconditioner<C,P> >;
  public:
    //! \brief The domain type of the preconditioner.
    typedef typename P::domain_type domain_type;
    //! \brief The range type of the preconditioner.
    typedef typename P::range_type range_type;
    //! \brief The type of the communication object.
    typedef C communication_type;

    // define the category
    enum {
      //! \brief The category the preconditioner is part of.
      category=SolverCategory::nonoverlapping
    };

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param prec The sequential preconditioner.
       \param c The communication object for syncing owner and copy
       data points. (E.~g. OwnerOverlapCommunication )
     */
    NonoverlappingBlockPreconditioner (P& prec, const communication_type& c)
      : preconditioner(prec), communication(c)
    {}

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(domain_type&,range_type&)
     */
    virtual void pre (domain_type& x, range_type& b)
    {
      preconditioner.pre(x,b);
    }

    /*!
       \brief Apply the preconditioner

       \copydoc Preconditioner::apply(domain_type&,const range_type&)
     */
    virtual void apply (domain_type& v, const range_type& d)
    {
      preconditioner.apply(v,d);
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(domain_type&)
     */
    virtual void post (domain_type& x)
    {
      preconditioner.post(x);
    }

  private:
    //! \brief a sequential preconditioner
    P& preconditioner;

    //! \brief the communication object
    const communication_type& communication;
  };

  /** @} end documentation */

} // end namespace

#endif
