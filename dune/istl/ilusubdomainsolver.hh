// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ILUSUBDOMAINSOLVER_HH
#define DUNE_ISTL_ILUSUBDOMAINSOLVER_HH

#include <map>
#include <dune/common/typetraits.hh>
#include <dune/istl/preconditioners.hh>
#include "matrix.hh"
#include <cmath>
#include <cstdlib>

namespace Dune {

  /**
   * @file
   * @brief Various local subdomain solvers based on ILU
   * for SeqOverlappingSchwarz.
   * @author Markus Blatt
   */
  /**
   * @addtogroup ISTL
   * @{
   */
  /**
   * @brief base class encapsulating common algorithms of ILU0SubdomainSolver
   * and ILUNSubdomainSolver.
   * @tparam M The type of the matrix.
   * @tparam X The type of the vector for the domain.
   * @tparam X The type of the vector for the range.
   *
   */
  template<class M, class X, class Y>
  class ILUSubdomainSolver  {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;

    /**
     * @brief Apply the subdomain solver.
     *
     * On entry v=? and d=b-A(x) (although this might not be
     * computed in that way. On exit v contains the update
     */
    virtual void apply (X& v, const Y& d) =0;

    virtual ~ILUSubdomainSolver()
    {}

  protected:
    /**
     * @brief Copy the local part of the global matrix to ILU.
     * @param A The global matrix.
     * @param rowset The global indices of the local problem.
     */
    template<class S>
    std::size_t copyToLocalMatrix(const M& A, S& rowset);

    //! \brief The ILU0 decomposition of the matrix, or the local matrix
    // for ILUN
    matrix_type ILU;
  };

  /**
   * @brief Exact subdomain solver using ILU(p) with appropriate p.
   * @tparam M The type of the matrix.
   * @tparam X The type of the vector for the domain.
   * @tparam X The type of the vector for the range.
   */
  template<class M, class X, class Y>
  class ILU0SubdomainSolver
    : public ILUSubdomainSolver<M,X,Y>{
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    typedef typename std::remove_const<M>::type rilu_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;


    /**
     * @brief Apply the subdomain solver.
     * @copydoc ILUSubdomainSolver::apply
     */
    void apply (X& v, const Y& d)
    {
      ILU::blockILUBacksolve(this->ILU,v,d);
    }
    /**
     * @brief Set the data of the local problem.
     *
     * @param A The global matrix.
     * @param rowset The global indices of the local problem.
     * @tparam S The type of the set with the indices.
     */
    template<class S>
    void setSubMatrix(const M& A, S& rowset);

  };

  template<class M, class X, class Y>
  class ILUNSubdomainSolver
    : public ILUSubdomainSolver<M,X,Y>{
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    typedef typename std::remove_const<M>::type rilu_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;

    /**
     * @brief Apply the subdomain solver.
     * @copydoc ILUSubdomainSolver::apply
     */
    void apply (X& v, const Y& d)
    {
      ILU::blockILUBacksolve(RILU,v,d);
    }

    /**
     * @brief Set the data of the local problem.
     *
     * @param A The global matrix.
     * @param rowset The global indices of the local problem.
     * @tparam S The type of the set with the indices.
     */
    template<class S>
    void setSubMatrix(const M& A, S& rowset);

  private:
    /**
     * @brief Storage for the ILUN decomposition.
     */
    rilu_type RILU;
  };



  template<class M, class X, class Y>
  template<class S>
  std::size_t ILUSubdomainSolver<M,X,Y>::copyToLocalMatrix(const M& A, S& rowSet)
  {
    // Calculate consecutive indices for local problem
    // while perserving the ordering
    typedef typename M::size_type size_type;
    typedef std::map<typename S::value_type,size_type> IndexMap;
    typedef typename IndexMap::iterator IMIter;
    IndexMap indexMap;
    IMIter guess = indexMap.begin();
    size_type localIndex=0;

    typedef typename S::const_iterator SIter;
    for(SIter rowIdx = rowSet.begin(), rowEnd=rowSet.end();
        rowIdx!= rowEnd; ++rowIdx, ++localIndex)
      guess = indexMap.insert(guess,
                              std::make_pair(*rowIdx,localIndex));


    // Build Matrix for local subproblem
    ILU.setSize(rowSet.size(),rowSet.size());
    ILU.setBuildMode(matrix_type::row_wise);

    // Create sparsity pattern
    typedef typename matrix_type::CreateIterator CIter;
    CIter rowCreator = ILU.createbegin();
    std::size_t offset=0;
    for(SIter rowIdx = rowSet.begin(), rowEnd=rowSet.end();
        rowIdx!= rowEnd; ++rowIdx, ++rowCreator) {
      // See which row entries are in our subset and add them to
      // the sparsity pattern
      guess = indexMap.begin();

      for(typename matrix_type::ConstColIterator col=A[*rowIdx].begin(),
          endcol=A[*rowIdx].end(); col != endcol; ++col) {
        // search for the entry in the row set
        guess = indexMap.find(col.index());
        if(guess!=indexMap.end()) {
          // add local index to row
          rowCreator.insert(guess->second);
          offset=std::max(offset,(std::size_t)std::abs((int)(guess->second-rowCreator.index())));
        }
      }

    }

    // Insert the matrix values for the local problem
    typename matrix_type::iterator iluRow=ILU.begin();

    for(SIter rowIdx = rowSet.begin(), rowEnd=rowSet.end();
        rowIdx!= rowEnd; ++rowIdx, ++iluRow) {
      // See which row entries are in our subset and add them to
      // the sparsity pattern
      typename matrix_type::ColIterator localCol=iluRow->begin();
      for(typename matrix_type::ConstColIterator col=A[*rowIdx].begin(),
          endcol=A[*rowIdx].end(); col != endcol; ++col) {
        // search for the entry in the row set
        guess = indexMap.find(col.index());
        if(guess!=indexMap.end()) {
          // set local value
          (*localCol)=(*col);
          ++localCol;
        }
      }
    }
    return offset;
  }


  template<class M, class X, class Y>
  template<class S>
  void ILU0SubdomainSolver<M,X,Y>::setSubMatrix(const M& A, S& rowSet)
  {
    this->copyToLocalMatrix(A,rowSet);
    ILU::blockILU0Decomposition(this->ILU);
  }

  template<class M, class X, class Y>
  template<class S>
  void ILUNSubdomainSolver<M,X,Y>::setSubMatrix(const M& A, S& rowSet)
  {
    std::size_t offset=copyToLocalMatrix(A,rowSet);
    RILU.setSize(rowSet.size(),rowSet.size(), (1+2*offset)*rowSet.size());
    RILU.setBuildMode(matrix_type::row_wise);
    ILU::blockILUDecomposition(this->ILU, (offset+1)/2, RILU);
  }

  /** @} */
} // end name space DUNE


#endif
