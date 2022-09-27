// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_NOVLPSCHWARZ_HH
#define DUNE_ISTL_NOVLPSCHWARZ_HH

#include <iostream>              // for input/output to shell
#include <fstream>               // for input/output to files
#include <vector>                // STL vector class
#include <sstream>

#include <cmath>                // Yes, we do some math here

#include <dune/common/timer.hh>

#include <dune/common/hybridutilities.hh>

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

    typedef typename C::PIS PIS;
    typedef typename C::RI RI;
    typedef typename RI::RemoteIndexList RIL;
    typedef typename RI::const_iterator RIIterator;
    typedef typename RIL::const_iterator RILIterator;
    typedef typename M::ConstColIterator ColIterator;
    typedef typename M::ConstRowIterator RowIterator;
    typedef std::multimap<int,int> MM;
    typedef std::multimap<int,std::pair<int,RILIterator> > RIMap;
    typedef typename RIMap::iterator RIMapit;

    /**
     * @brief constructor: just store a reference to a matrix.
     *
     * @param A The assembled matrix.
     * @param com The communication object for syncing owner and copy
     * data points. (E.~g. OwnerOverlapCommunication )
     */
    NonoverlappingSchwarzOperator (const matrix_type& A, const communication_type& com)
      : _A_(stackobject_to_shared_ptr(A)), communication(com), buildcomm(true)
    {}

    NonoverlappingSchwarzOperator (std::shared_ptr<const matrix_type> A, const communication_type& com)
      : _A_(A), communication(com), buildcomm(true)
    {}

    //! apply operator to x:  \f$ y = A(x) \f$
    virtual void apply (const X& x, Y& y) const
    {
      y = 0;
      novlp_op_apply(x,y,1);
      communication.addOwnerCopyToOwnerCopy(y,y);
    }

    //! apply operator to x, scale and add:  \f$ y = y + \alpha A(x) \f$
    virtual void applyscaleadd (field_type alpha, const X& x, Y& y) const
    {
      // only apply communication to alpha*A*x to make it consistent,
      // y already has to be consistent.
      Y y1(y);
      y = 0;
      novlp_op_apply(x,y,alpha);
      communication.addOwnerCopyToOwnerCopy(y,y);
      y += y1;
    }

    //! get matrix via *
    virtual const matrix_type& getmat () const
    {
      return *_A_;
    }

    void novlp_op_apply (const X& x, Y& y, field_type alpha) const
    {
      //get index sets
      const PIS& pis=communication.indexSet();
      const RI& ri = communication.remoteIndices();

      // at the beginning make a multimap "bordercontribution".
      // process has i and j as border dofs but is not the owner
      // => only contribute to Ax if i,j is in bordercontribution
      if (buildcomm == true) {

        // set up mask vector
        if (mask.size()!=static_cast<typename std::vector<double>::size_type>(x.size())) {
          mask.resize(x.size());
          for (typename std::vector<double>::size_type i=0; i<mask.size(); i++)
            mask[i] = 1;
          for (typename PIS::const_iterator i=pis.begin(); i!=pis.end(); ++i)
            if (i->local().attribute()==OwnerOverlapCopyAttributeSet::copy)
              mask[i->local().local()] = 0;
            else if (i->local().attribute()==OwnerOverlapCopyAttributeSet::overlap)
              mask[i->local().local()] = 2;
        }

        for (MM::iterator iter = bordercontribution.begin();
             iter != bordercontribution.end(); ++iter)
          bordercontribution.erase(iter);
        std::map<int,int> owner; //key: local index i, value: process, that owns i
        RIMap rimap;

        // for each local index make multimap rimap:
        // key: local index i, data: pair of process that knows i and pointer to RI entry
        for (RowIterator i = _A_->begin(); i != _A_->end(); ++i)
          if (mask[i.index()] == 0)
            for (RIIterator remote = ri.begin(); remote != ri.end(); ++remote) {
              RIL& ril = *(remote->second.first);
              for (RILIterator rindex = ril.begin(); rindex != ril.end(); ++rindex)
                if (rindex->attribute() != OwnerOverlapCopyAttributeSet::overlap)
                  if (rindex->localIndexPair().local().local() == i.index()) {
                    rimap.insert
                      (std::make_pair(i.index(),
                                      std::pair<int,RILIterator>(remote->first, rindex)));
                    if(rindex->attribute()==OwnerOverlapCopyAttributeSet::owner)
                      owner.insert(std::make_pair(i.index(),remote->first));
                  }
            }

        int iowner = 0;
        for (RowIterator i = _A_->begin(); i != _A_->end(); ++i) {
          if (mask[i.index()] == 0) {
            std::map<int,int>::iterator it = owner.find(i.index());
            iowner = it->second;
            std::pair<RIMapit, RIMapit> foundiit = rimap.equal_range(i.index());
            for (ColIterator j = (*_A_)[i.index()].begin(); j != (*_A_)[i.index()].end(); ++j) {
              if (mask[j.index()] == 0) {
                bool flag = true;
                for (RIMapit foundi = foundiit.first; foundi != foundiit.second; ++foundi) {
                  std::pair<RIMapit, RIMapit> foundjit = rimap.equal_range(j.index());
                  for (RIMapit foundj = foundjit.first; foundj != foundjit.second; ++foundj)
                    if (foundj->second.first == foundi->second.first)
                      if (foundj->second.second->attribute() == OwnerOverlapCopyAttributeSet::owner
                          || foundj->second.first == iowner
                          || foundj->second.first  < communication.communicator().rank()) {
                        flag = false;
                        continue;
                      }
                  if (flag == false)
                    continue;
                }
                // don´t contribute to Ax if
                // 1. the owner of j has i as interior/border dof
                // 2. iowner has j as interior/border dof
                // 3. there is another process with smaller rank that has i and j
                // as interor/border dofs
                // if the owner of j does not have i as interior/border dof,
                // it will not be taken into account
                if (flag==true)
                  bordercontribution.insert(std::pair<int,int>(i.index(),j.index()));
              }
            }
          }
        }
        buildcomm = false;
      }

      //compute alpha*A*x nonoverlapping case
      for (RowIterator i = _A_->begin(); i != _A_->end(); ++i) {
        if (mask[i.index()] == 0) {
          //dof doesn't belong to process but is border (not ghost)
          for (ColIterator j = (*_A_)[i.index()].begin(); j != (*_A_)[i.index()].end(); ++j) {
            if (mask[j.index()] == 1) //j is owner => then sum entries
              Impl::asMatrix(*j).usmv(alpha,x[j.index()],y[i.index()]);
            else if (mask[j.index()] == 0) {
              std::pair<MM::iterator, MM::iterator> itp =
                bordercontribution.equal_range(i.index());
              for (MM::iterator it = itp.first; it != itp.second; ++it)
                if ((*it).second == (int)j.index())
                  Impl::asMatrix(*j).usmv(alpha,x[j.index()],y[i.index()]);
            }
          }
        }
        else if (mask[i.index()] == 1) {
          for (ColIterator j = (*_A_)[i.index()].begin(); j != (*_A_)[i.index()].end(); ++j)
            if (mask[j.index()] != 2)
              Impl::asMatrix(*j).usmv(alpha,x[j.index()],y[i.index()]);
        }
      }
    }

    //! Category of the linear operator (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::nonoverlapping;
    }

    //! Get the object responsible for communication
    const communication_type& getCommunication() const
    {
      return communication;
    }
  private:
    std::shared_ptr<const matrix_type> _A_;
    const communication_type& communication;
    mutable bool buildcomm;
    mutable std::vector<double> mask;
    mutable std::multimap<int,int>  bordercontribution;
  };

  /** @} */

  namespace Amg
  {
    template<class T> struct ConstructionTraits;
  }

  /**
   * @addtogroup ISTL_Prec
   * @{
   */

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
    : public Preconditioner<typename P::domain_type,typename P::range_type> {
    friend struct Amg::ConstructionTraits<NonoverlappingBlockPreconditioner<C,P> >;
    using X = typename P::domain_type;
    using Y = typename P::range_type;
  public:
    //! \brief The domain type of the preconditioner.
    typedef typename P::domain_type domain_type;
    //! \brief The range type of the preconditioner.
    typedef typename P::range_type range_type;
    //! \brief The type of the communication object.
    typedef C communication_type;

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param prec The sequential preconditioner.
       \param c The communication object for syncing owner and copy
       data points. (E.~g. OwnerOverlapCommunication )
     */
    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param p The sequential preconditioner.
       \param c The communication object for syncing overlap and copy
       data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    NonoverlappingBlockPreconditioner (P& p, const communication_type& c)
      : _preconditioner(stackobject_to_shared_ptr(p)), _communication(c)
    {   }

    /*! \brief Constructor.

       constructor gets all parameters to operate the prec.
       \param p The sequential preconditioner.
       \param c The communication object for syncing overlap and copy
       data points. (E.~g. OwnerOverlapCopyCommunication )
     */
    NonoverlappingBlockPreconditioner (const std::shared_ptr<P>& p, const communication_type& c)
      : _preconditioner(p), _communication(c)
    {   }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(domain_type&,range_type&)
     */
    virtual void pre (domain_type& x, range_type& b)
    {
      _preconditioner->pre(x,b);
    }

    /*!
       \brief Apply the preconditioner

       \copydoc Preconditioner::apply(domain_type&,const range_type&)
     */
    virtual void apply (domain_type& v, const range_type& d)
    {
      // block preconditioner equivalent to WrappedPreconditioner from
      // pdelab/backend/ovlpistsolverbackend.hh,
      // but not to BlockPreconditioner from schwarz.hh
      _preconditioner->apply(v,d);
      _communication.addOwnerCopyToOwnerCopy(v,v);
    }

    template<bool forward>
    void apply (X& v, const Y& d)
    {
      _preconditioner->template apply<forward>(v,d);
      _communication.addOwnerCopyToOwnerCopy(v,v);
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(domain_type&)
     */
    virtual void post (domain_type& x)
    {
      _preconditioner->post(x);
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::nonoverlapping;
    }

  private:
    //! \brief a sequential preconditioner
    std::shared_ptr<P> _preconditioner;

    //! \brief the communication object
    const communication_type& _communication;
  };

  /** @} end documentation */

} // end namespace

#endif
