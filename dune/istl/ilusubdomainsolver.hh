// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ILUSUBDOMAIN_HH
#define DUNE_ISTL_ILUSUBDOMAIN_HH

#include <map>
#include <dune/common/typetraits.hh>

namespace Dune {

  template<class M, class X, class Y>
  class ILU0SubdomainSolver  {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename Dune::remove_const<M>::type matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    typedef typename X::field_type field_type;

    void apply (X& v, const Y& d)
    {
      bilu_backsolve(ILU,v,d);
    }

    template<class S>
    void setSubMatrix(const M& A, S& rowset);

  private:
    //! \brief The relaxation factor to use.
    field_type _w;
    //! \brief The ILU0 decomposition of the matrix.
    matrix_type ILU;
  };

  template<class M, class X, class Y>
  template<class S>
  void ILU0SubdomainSolver<M,X,Y>::setSubMatrix(const M& A, S& rowSet)
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
    typedef typename S::const_iterator RIter;
    for(SIter rowIdx = rowSet.begin(), rowEnd=rowSet.end();
        rowIdx!= rowEnd; ++rowIdx, ++rowCreator) {
      // See wich row entries are in our subset and add them to
      // the sparsity pattern
      guess = indexMap.begin();
      for(typename matrix_type::ConstColIterator col=A[*rowIdx].begin(),
          endcol=A[*rowIdx].end(); col != endcol; ++col) {
        // search for the entry in the row set
        guess = indexMap.find(col.index());
        if(guess!=indexMap.end())
          // add local index to row
          rowCreator.insert(guess->second);
      }
    }

    // Insert the matrix values for the local problem
    typename matrix_type::iterator iluRow=ILU.begin();

    for(SIter rowIdx = rowSet.begin(), rowEnd=rowSet.end();
        rowIdx!= rowEnd; ++rowIdx, ++iluRow) {
      // See wich row entries are in our subset and add them to
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
    bilu0_decomposition(ILU);
  }

} // end name space DUNE


#endif
