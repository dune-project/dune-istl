// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_MATRIX_UTILS_HH
#define DUNE_MATRIX_UTILS_HH

#include <set>
#include <limits>
#include <dune/common/typetraits.hh>
#include <dune/common/static_assert.hh>
#include "istlexception.hh"

namespace Dune
{
  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /**
   * @file
   * @brief Some handy generic functions for ISTL matrices.
   * @author Markus Blatt
   */
  namespace
  {

    template<int i>
    struct NonZeroCounter
    {
      template<class M>
      static typename M::size_type count(const M& matrix)
      {
        typedef typename M::ConstRowIterator RowIterator;

        RowIterator endRow = matrix.end();
        typename M::size_type nonZeros = 0;

        for(RowIterator row = matrix.begin(); row != endRow; ++row) {
          typedef typename M::ConstColIterator Entry;
          Entry endEntry = row->end();
          for(Entry entry = row->begin(); entry != endEntry; ++entry) {
            nonZeros += NonZeroCounter<i-1>::count(*entry);
          }
        }
        return nonZeros;
      }
    };

    template<>
    struct NonZeroCounter<1>
    {
      template<class M>
      static typename M::size_type count(const M& matrix)
      {
        return matrix.N()*matrix.M();
      }
    };

  }

  /**
   * @brief Check whether the a matrix has diagonal values
   * on blocklevel recursion levels.
   */
  template<std::size_t blocklevel, std::size_t l=blocklevel>
  struct CheckIfDiagonalPresent
  {
    /**
     * @brief Check whether the a matrix has diagonal values
     * on blocklevel recursion levels.
     */
    template<class Matrix>
    static void check(const Matrix& mat)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      CheckIfDiagonalPresent<blocklevel-1,l>::check(mat);
#endif
    }
  };

  template<std::size_t l>
  struct CheckIfDiagonalPresent<0,l>
  {
    template<class Matrix>
    static void check(const Matrix& mat)
    {
      typedef typename Matrix::ConstRowIterator Row;
      for(Row row = mat.begin(); row!=mat.end(); ++row) {
        if(row->find(row.index())==row->end())
          DUNE_THROW(ISTLError, "Missing diagonal value in row "<<row.index()
                                                                <<" at block recursion level "<<l);
      }
    }
  };

  /**
   * @brief Get the number of nonzero fields in the matrix.
   *
   * This is not the number of nonzero blocks, but the number of non
   * zero scalar entries (on blocklevel 1) if the matrix is viewed as
   * a flat matrix.
   *
   * For FieldMatrix this is simply the number of columns times the
   * number of rows, for a BCRSMatrix<FieldMatrix<K,n,m>> this is the
   * number of nonzero blocks time n*m.
   */
  template<class M>
  inline int countNonZeros(const M& matrix)
  {
    return NonZeroCounter<M::blocklevel>::count(matrix);
  }
  /*
     template<class M>
     struct ProcessOnFieldsOfMatrix
   */

  /** @} */
  namespace
  {
    struct CompPair {
      template<class G,class M>
      bool operator()(const std::pair<G,M>& p1, const std::pair<G,M>& p2)
      {
        return p1.first<p2.first;
      }
    };

  }
  template<class M, class C>
  void printGlobalSparseMatrix(const M& mat, C& ooc, std::ostream& os)
  {
    typedef typename C::ParallelIndexSet::const_iterator IIter;
    typedef typename C::OwnerSet OwnerSet;
    typedef typename C::ParallelIndexSet::GlobalIndex GlobalIndex;

    GlobalIndex gmax=0;

    for(IIter idx=ooc.indexSet().begin(), eidx=ooc.indexSet().end();
        idx!=eidx; ++idx)
      gmax=std::max(gmax,idx->global());

    gmax=ooc.communicator().max(gmax);
    ooc.buildGlobalLookup();

    for(IIter idx=ooc.indexSet().begin(), eidx=ooc.indexSet().end();
        idx!=eidx; ++idx) {
      if(OwnerSet::contains(idx->local().attribute()))
      {
        typedef typename  M::block_type Block;

        std::set<std::pair<GlobalIndex,Block>,CompPair> entries;

        // sort rows
        typedef typename M::ConstColIterator CIter;
        for(CIter c=mat[idx->local()].begin(), cend=mat[idx->local()].end();
            c!=cend; ++c) {
          const typename C::ParallelIndexSet::IndexPair* pair
            =ooc.globalLookup().pair(c.index());
          assert(pair);
          entries.insert(std::make_pair(pair->global(), *c));
        }

        //wait until its the rows turn.
        GlobalIndex rowidx = idx->global();
        GlobalIndex cur=std::numeric_limits<GlobalIndex>::max();
        while(cur!=rowidx)
          cur=ooc.communicator().min(rowidx);

        // print rows
        typedef typename std::set<std::pair<GlobalIndex,Block> >::iterator SIter;
        for(SIter s=entries.begin(), send=entries.end(); s!=send; ++s)
          os<<idx->global()<<" "<<s->first<<" "<<s->second<<std::endl;


      }
    }

    ooc.freeGlobalLookup();
    // Wait until everybody is finished
    GlobalIndex cur=std::numeric_limits<GlobalIndex>::max();
    while(cur!=ooc.communicator().min(cur)) ;
  }

}
#endif
