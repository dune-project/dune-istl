// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIXUTILS_HH
#define DUNE_ISTL_MATRIXUTILS_HH

#include <set>
#include <vector>
#include <limits>
#include <dune/common/typetraits.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/istl/scaledidmatrix.hh>
#include "istlexception.hh"

namespace Dune
{

#ifndef DOYXGEN
  template<typename B, typename A>
  class BCRSMatrix;

  template<typename K, int n, int m>
  class FieldMatrix;

  template<class T, class A>
  class Matrix;
#endif

  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /**
   * @file
   * @brief Some handy generic functions for ISTL matrices.
   * @author Markus Blatt
   */
  /**
   * @brief Check whether the a matrix has diagonal values
   * on blocklevel recursion levels.
   */
  template<class Matrix, std::size_t blocklevel, std::size_t l=blocklevel>
  struct CheckIfDiagonalPresent
  {
    /**
     * @brief Check whether the a matrix has diagonal values
     * on blocklevel recursion levels.
     */
    static void check([[maybe_unused]] const Matrix& mat)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      typedef typename Matrix::ConstRowIterator Row;
      typedef typename Matrix::ConstColIterator Entry;
      for(Row row = mat.begin(); row!=mat.end(); ++row) {
        Entry diagonal = row->find(row.index());
        if(diagonal==row->end())
          DUNE_THROW(ISTLError, "Missing diagonal value in row "<<row.index()
                                                                <<" at block recursion level "<<l-blocklevel);
        else{
          auto m = Impl::asMatrix(*diagonal);
          CheckIfDiagonalPresent<decltype(m),blocklevel-1,l>::check(m);
        }
      }
#endif
    }
  };

  template<class Matrix, std::size_t l>
  struct CheckIfDiagonalPresent<Matrix,0,l>
  {
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

  template<typename FirstRow, typename... Args>
  class MultiTypeBlockMatrix;

  template<std::size_t blocklevel, std::size_t l, typename T1, typename... Args>
  struct CheckIfDiagonalPresent<MultiTypeBlockMatrix<T1,Args...>,
      blocklevel,l>
  {
    typedef MultiTypeBlockMatrix<T1,Args...> Matrix;

    /**
     * @brief Check whether the a matrix has diagonal values
     * on blocklevel recursion levels.
     */
    static void check(const Matrix& /* mat */)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      // TODO Implement check
#endif
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
  inline auto countNonZeros(const M&,
                            [[maybe_unused]] typename std::enable_if_t<Dune::IsNumber<M>::value>* sfinae = nullptr)
  {
    return 1;
  }

  template<class M>
  inline auto countNonZeros(const M& matrix,
                            [[maybe_unused]] typename std::enable_if_t<!Dune::IsNumber<M>::value>* sfinae = nullptr)
  {
    typename M::size_type nonZeros = 0;
    for(auto&& row : matrix)
      for(auto&& entry : row)
        nonZeros += countNonZeros(entry);
    return nonZeros;
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
      bool operator()(const std::pair<G,M>& p1, const std::pair<G,M>& p2) const
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
        typedef typename std::set<std::pair<GlobalIndex,Block>,CompPair>::iterator SIter;
        for(SIter s=entries.begin(), send=entries.end(); s!=send; ++s)
          os<<idx->global()<<" "<<s->first<<" "<<s->second<<std::endl;


      }
    }

    ooc.freeGlobalLookup();
    // Wait until everybody is finished
    GlobalIndex cur=std::numeric_limits<GlobalIndex>::max();
    while(cur!=ooc.communicator().min(cur)) ;
  }

  // Default implementation for scalar types
  template<typename M>
  struct MatrixDimension
  {
    static_assert(IsNumber<M>::value, "MatrixDimension is not implemented for this type!");

    static auto rowdim(const M& A)
    {
      return 1;
    }

    static auto coldim(const M& A)
    {
      return 1;
    }
  };

  // Default implementation for scalar types
  template<typename B, typename TA>
  struct MatrixDimension<Matrix<B,TA> >
  {
    using block_type = typename Matrix<B,TA>::block_type;
    using size_type = typename Matrix<B,TA>::size_type;

    static size_type rowdim (const Matrix<B,TA>& A, size_type i)
    {
      return MatrixDimension<block_type>::rowdim(A[i][0]);
    }

    static size_type coldim (const Matrix<B,TA>& A, size_type c)
    {
      return MatrixDimension<block_type>::coldim(A[0][c]);
    }

    static size_type rowdim (const Matrix<B,TA>& A)
    {
      size_type nn=0;
      for (size_type i=0; i<A.N(); i++)
        nn += rowdim(A,i);
      return nn;
    }

    static size_type coldim (const Matrix<B,TA>& A)
    {
      size_type nn=0;
      for (size_type i=0; i<A.M(); i++)
        nn += coldim(A,i);
      return nn;
    }
  };


  template<typename B, typename TA>
  struct MatrixDimension<BCRSMatrix<B,TA> >
  {
    typedef BCRSMatrix<B,TA> Matrix;
    typedef typename Matrix::block_type block_type;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim (const Matrix& A, size_type i)
    {
      const B* row = A.r[i].getptr();
      if(row)
        return MatrixDimension<block_type>::rowdim(*row);
      else
        return 0;
    }

    static size_type coldim (const Matrix& A, size_type c)
    {
      // find an entry in column c
      if (A.nnz_ > 0)
      {
        for (size_type k=0; k<A.nnz_; k++) {
          if (A.j_.get()[k] == c) {
            return MatrixDimension<block_type>::coldim(A.a[k]);
          }
        }
      }
      else
      {
        for (size_type i=0; i<A.N(); i++)
        {
          size_type* j = A.r[i].getindexptr();
          B*   a = A.r[i].getptr();
          for (size_type k=0; k<A.r[i].getsize(); k++)
            if (j[k]==c) {
              return MatrixDimension<block_type>::coldim(a[k]);
            }
        }
      }

      // not found
      return 0;
    }

    static size_type rowdim (const Matrix& A){
      size_type nn=0;
      for (size_type i=0; i<A.N(); i++)
        nn += rowdim(A,i);
      return nn;
    }

    static size_type coldim (const Matrix& A){
      typedef typename Matrix::ConstRowIterator ConstRowIterator;
      typedef typename Matrix::ConstColIterator ConstColIterator;

      // The following code has a complexity of nnz, and
      // typically a very small constant.
      //
      std::vector<size_type> coldims(A.M(),
                                     std::numeric_limits<size_type>::max());

      for (ConstRowIterator row=A.begin(); row!=A.end(); ++row)
        for (ConstColIterator col=row->begin(); col!=row->end(); ++col)
          // only compute blocksizes we don't already have
          if (coldims[col.index()]==std::numeric_limits<size_type>::max())
            coldims[col.index()] = MatrixDimension<block_type>::coldim(*col);

      size_type sum = 0;
      for (typename std::vector<size_type>::iterator it=coldims.begin();
           it!=coldims.end(); ++it)
        // skip rows for which no coldim could be determined
        if ((*it)>=0)
          sum += *it;

      return sum;
    }
  };


  template<typename B, int n, int m, typename TA>
  struct MatrixDimension<BCRSMatrix<FieldMatrix<B,n,m> ,TA> >
  {
    typedef BCRSMatrix<FieldMatrix<B,n,m> ,TA> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim (const Matrix& /*A*/, size_type /*i*/)
    {
      return n;
    }

    static size_type coldim (const Matrix& /*A*/, size_type /*c*/)
    {
      return m;
    }

    static size_type rowdim (const Matrix& A) {
      return A.N()*n;
    }

    static size_type coldim (const Matrix& A) {
      return A.M()*m;
    }
  };

  template<typename K, int n, int m>
  struct MatrixDimension<FieldMatrix<K,n,m> >
  {
    typedef FieldMatrix<K,n,m> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type coldim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type rowdim(const Matrix& /*A*/)
    {
      return n;
    }

    static size_type coldim(const Matrix& /*A*/)
    {
      return m;
    }
  };

  template <class T>
  struct MatrixDimension<Dune::DynamicMatrix<T> >
  {
    typedef Dune::DynamicMatrix<T> MatrixType;
    typedef typename MatrixType::size_type size_type;

    static size_type rowdim(const MatrixType& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type coldim(const MatrixType& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type rowdim(const MatrixType& A)
    {
      return A.N();
    }

    static size_type coldim(const MatrixType& A)
    {
      return A.M();
    }
  };

  template<typename K, int n, int m, typename TA>
  struct MatrixDimension<Matrix<FieldMatrix<K,n,m>, TA> >
  {
    typedef Matrix<FieldMatrix<K,n,m>, TA> ThisMatrix;
    typedef typename ThisMatrix::size_type size_type;

    static size_type rowdim(const ThisMatrix& /*A*/, size_type /*r*/)
    {
      return n;
    }

    static size_type coldim(const ThisMatrix& /*A*/, size_type /*r*/)
    {
      return m;
    }

    static size_type rowdim(const ThisMatrix& A)
    {
      return A.N()*n;
    }

    static size_type coldim(const ThisMatrix& A)
    {
      return A.M()*m;
    }
  };

  template<typename K, int n>
  struct MatrixDimension<DiagonalMatrix<K,n> >
  {
    typedef DiagonalMatrix<K,n> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type coldim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type rowdim(const Matrix& /*A*/)
    {
      return n;
    }

    static size_type coldim(const Matrix& /*A*/)
    {
      return n;
    }
  };

  template<typename K, int n>
  struct MatrixDimension<ScaledIdentityMatrix<K,n> >
  {
    typedef ScaledIdentityMatrix<K,n> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type coldim(const Matrix& /*A*/, size_type /*r*/)
    {
      return 1;
    }

    static size_type rowdim(const Matrix& /*A*/)
    {
      return n;
    }

    static size_type coldim(const Matrix& /*A*/)
    {
      return n;
    }
  };

  /**
   * @brief Test whether a type is an ISTL Matrix
   */
  template<typename T>
  struct IsMatrix
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = false
    };
  };

  template<typename T>
  struct IsMatrix<DenseMatrix<T> >
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = true
    };
  };


  template<typename T, typename A>
  struct IsMatrix<BCRSMatrix<T,A> >
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = true
    };
  };

  template<typename T>
  struct PointerCompare
  {
    bool operator()(const T* l, const T* r)
    {
      return *l < *r;
    }
  };

}
#endif
