// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ILU_HH
#define DUNE_ISTL_ILU_HH

#include <cmath>
#include <complex>
#include <map>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>

#include "istlexception.hh"

/** \file
 * \brief  The incomplete LU factorization kernels
 */

namespace Dune {

  /** @addtogroup ISTL_Kernel
          @{
   */

  namespace ILU {

    //! compute ILU decomposition of A. A is overwritten by its decomposition
    template<class M>
    void blockILU0Decomposition (M& A)
    {
      // iterator types
      typedef typename M::RowIterator rowiterator;
      typedef typename M::ColIterator coliterator;
      typedef typename M::block_type block;

      // implement left looking variant with stored inverse
      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        // coliterator is diagonal after the following loop
        coliterator endij=(*i).end();           // end of row i
        coliterator ij;

        // eliminate entries left of diagonal; store L factor
        for (ij=(*i).begin(); ij.index()<i.index(); ++ij)
        {
          // find A_jj which eliminates A_ij
          coliterator jj = A[ij.index()].find(ij.index());

          // compute L_ij = A_jj^-1 * A_ij
          Impl::asMatrix(*ij).rightmultiply(Impl::asMatrix(*jj));

          // modify row
          coliterator endjk=A[ij.index()].end();                 // end of row j
          coliterator jk=jj; ++jk;
          coliterator ik=ij; ++ik;
          while (ik!=endij && jk!=endjk)
            if (ik.index()==jk.index())
            {
              block B(*jk);
              Impl::asMatrix(B).leftmultiply(Impl::asMatrix(*ij));
              *ik -= B;
              ++ik; ++jk;
            }
            else
            {
              if (ik.index()<jk.index())
                ++ik;
              else
                ++jk;
            }
        }

        // invert pivot and store it in A
        if (ij.index()!=i.index())
          DUNE_THROW(ISTLError,"diagonal entry missing");
        try {
          Impl::asMatrix(*ij).invert();   // compute inverse of diagonal block
        }
        catch (Dune::FMatrixError & e) {
          DUNE_THROW(MatrixBlockError, "ILU failed to invert matrix block A["
                     << i.index() << "][" << ij.index() << "]" << e.what();
                     th__ex.r=i.index(); th__ex.c=ij.index(););
        }
      }
    }

    //! LU backsolve with stored inverse
    template<class M, class X, class Y>
    void blockILUBacksolve (const M& A, X& v, const Y& d)
    {
      // iterator types
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type dblock;
      typedef typename X::block_type vblock;

      // lower triangular solve
      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        // We need to be careful here: Directly using
        // auto rhs = Impl::asVector(d[ i.index() ]);
        // is not OK in case this is a proxy. Hence
        // we first have to copy the value. Notice that
        // this is still not OK, if the vector type itself returns
        // proxy references.
        dblock rhsValue(d[i.index()]);
        auto&& rhs = Impl::asVector(rhsValue);
        for (coliterator j=(*i).begin(); j.index()<i.index(); ++j)
          Impl::asMatrix(*j).mmv(Impl::asVector(v[j.index()]),rhs);
        Impl::asVector(v[i.index()]) = rhs;           // Lii = I
      }

      // upper triangular solve
      rowiterator rendi=A.beforeBegin();
      for (rowiterator i=A.beforeEnd(); i!=rendi; --i)
      {
        // We need to be careful here: Directly using
        // auto rhs = Impl::asVector(v[ i.index() ]);
        // is not OK in case this is a proxy. Hence
        // we first have to copy the value. Notice that
        // this is still not OK, if the vector type itself returns
        // proxy references.
        vblock rhsValue(v[i.index()]);
        auto&& rhs = Impl::asVector(rhsValue);
        coliterator j;
        for (j=(*i).beforeEnd(); j.index()>i.index(); --j)
          Impl::asMatrix(*j).mmv(Impl::asVector(v[j.index()]),rhs);
        auto&& vi = Impl::asVector(v[i.index()]);
        Impl::asMatrix(*j).mv(rhs,vi);           // diagonal stores inverse!
      }
    }

    // recursive function template to access first entry of a matrix
    template<class M>
    typename M::field_type& firstMatrixElement (M& A,
                                                [[maybe_unused]] typename std::enable_if_t<!Dune::IsNumber<M>::value>* sfinae = nullptr)
    {
      return firstMatrixElement(*(A.begin()->begin()));
    }

    template<class K>
    K& firstMatrixElement (K& A,
                           [[maybe_unused]] typename std::enable_if_t<Dune::IsNumber<K>::value>* sfinae = nullptr)
    {
      return A;
    }

    template<class K, int n, int m>
    K& firstMatrixElement (FieldMatrix<K,n,m>& A)
    {
      return A[0][0];
    }

    /*! ILU decomposition of order n
            Computes ILU decomposition of order n. The matrix ILU should
        be an empty matrix in row_wise creation mode. This allows the user
        to either specify the number of nonzero elements or to
            determine it automatically at run-time.
     */
    template<class M>
    void blockILUDecomposition (const M& A, int n, M& ILU)
    {
      // iterator types
      typedef typename M::ColIterator coliterator;
      typedef typename M::ConstRowIterator crowiterator;
      typedef typename M::ConstColIterator ccoliterator;
      typedef typename M::CreateIterator createiterator;
      typedef typename M::field_type K;
      typedef std::map<size_t, int> map;
      typedef typename map::iterator mapiterator;

      // symbolic factorization phase, store generation number in first matrix element
      crowiterator endi=A.end();
      createiterator ci=ILU.createbegin();
      for (crowiterator i=A.begin(); i!=endi; ++i)
      {
        map rowpattern; // maps column index to generation

        // initialize pattern with row of A
        for (ccoliterator j=(*i).begin(); j!=(*i).end(); ++j)
          rowpattern[j.index()] = 0;

        // eliminate entries in row which are to the left of the diagonal
        for (mapiterator ik=rowpattern.begin(); (*ik).first<i.index(); ++ik)
        {
          if ((*ik).second<n)
          {
            coliterator endk = ILU[(*ik).first].end();                       // end of row k
            coliterator kj = ILU[(*ik).first].find((*ik).first);                       // diagonal in k
            for (++kj; kj!=endk; ++kj)                       // row k eliminates in row i
            {
              // we misuse the storage to store an int. If the field_type is std::complex, we have to access the real/abs part
              // starting from C++11, we can use std::abs to always return a real value, even if it is double/float
              using std::abs;
              int generation = (int) Simd::lane(0, abs( firstMatrixElement(*kj) ));
              if (generation<n)
              {
                mapiterator ij = rowpattern.find(kj.index());
                if (ij==rowpattern.end())
                {
                  rowpattern[kj.index()] = generation+1;
                }
              }
            }
          }
        }

        // create row
        for (mapiterator ik=rowpattern.begin(); ik!=rowpattern.end(); ++ik)
          ci.insert((*ik).first);
        ++ci;           // now row i exist

        // write generation index into entries
        coliterator endILUij = ILU[i.index()].end();;
        for (coliterator ILUij=ILU[i.index()].begin(); ILUij!=endILUij; ++ILUij)
          Simd::lane(0, firstMatrixElement(*ILUij)) = (Simd::Scalar<K>) rowpattern[ILUij.index()];
      }

      // copy entries of A
      for (crowiterator i=A.begin(); i!=endi; ++i)
      {
        coliterator ILUij;
        coliterator endILUij = ILU[i.index()].end();;
        for (ILUij=ILU[i.index()].begin(); ILUij!=endILUij; ++ILUij)
          (*ILUij) = 0;           // clear row
        ccoliterator Aij = (*i).begin();
        ccoliterator endAij = (*i).end();
        ILUij = ILU[i.index()].begin();
        while (Aij!=endAij && ILUij!=endILUij)
        {
          if (Aij.index()==ILUij.index())
          {
            *ILUij = *Aij;
            ++Aij; ++ILUij;
          }
          else
          {
            if (Aij.index()<ILUij.index())
              ++Aij;
            else
              ++ILUij;
          }
        }
      }

      // call decomposition on pattern
      blockILU0Decomposition(ILU);
    }

    //! a simple compressed row storage matrix class
    template <class B, class Alloc = std::allocator<B>>
    struct CRS
    {
      typedef B       block_type;
      typedef size_t  size_type;

      CRS() : nRows_( 0 ) {}

      size_type rows() const { return nRows_; }

      size_type nonZeros() const
      {
        assert( rows_[ rows() ] != size_type(-1) );
        return rows_[ rows() ];
      }

      void resize( const size_type nRows )
      {
          if( nRows_ != nRows )
          {
            nRows_ = nRows ;
            rows_.resize( nRows_+1, size_type(-1) );
          }
      }

      void reserveAdditional( const size_type nonZeros )
      {
          const size_type needed = values_.size() + nonZeros ;
          if( values_.capacity() < needed )
          {
              const size_type estimate = needed * 1.1;
              values_.reserve( estimate );
              cols_.reserve( estimate );
          }
      }

      void push_back( const block_type& value, const size_type index )
      {
          values_.push_back( value );
          cols_.push_back( index );
      }

      std::vector< size_type  > rows_;
      std::vector< block_type, Alloc> values_;
      std::vector< size_type  > cols_;
      size_type nRows_;
    };

    //! convert ILU decomposition into CRS format for lower and upper triangular and inverse.
    template<class M, class CRS, class InvVector>
    void convertToCRS(const M& A, CRS& lower, CRS& upper, InvVector& inv )
    {
      typedef typename M :: size_type size_type;

      lower.resize( A.N() );
      upper.resize( A.N() );
      inv.resize( A.N() );

      // lower and upper triangular should store half of non zeros minus diagonal
      const size_t memEstimate = (A.nonzeroes() - A.N())/2;

      assert( A.nonzeroes() != 0 );
      lower.reserveAdditional( memEstimate );
      upper.reserveAdditional( memEstimate );

      const auto endi = A.end();
      size_type row = 0;
      size_type colcount = 0;
      lower.rows_[ 0 ] = colcount;
      for (auto i=A.begin(); i!=endi; ++i, ++row)
      {
        const size_type iIndex  = i.index();

        // store entries left of diagonal
        for (auto j=(*i).begin(); j.index() < iIndex; ++j )
        {
          lower.push_back( (*j), j.index() );
          ++colcount;
        }
        lower.rows_[ iIndex+1 ] = colcount;
      }

      const auto rendi = A.beforeBegin();
      row = 0;
      colcount = 0;
      upper.rows_[ 0 ] = colcount ;

      // NOTE: upper and inv store entries in reverse row and col order,
      // reverse here relative to ILU
      for (auto i=A.beforeEnd(); i!=rendi; --i, ++ row )
      {
        const auto endij=(*i).beforeBegin();    // end of row i

        const size_type iIndex = i.index();

        // store in reverse row order for faster access during backsolve
        for (auto j=(*i).beforeEnd(); j != endij; --j )
        {
          const size_type jIndex = j.index();
          if( j.index() == iIndex )
          {
            inv[ row ] = (*j);
            break; // assuming consecutive ordering of A
          }
          else if ( j.index() >= i.index() )
          {
            upper.push_back( (*j), jIndex );
            ++colcount ;
          }
        }
        upper.rows_[ row+1 ] = colcount;
      }
    } // end convertToCRS

    //! LU backsolve with stored inverse in CRS format for lower and upper triangular
    template<class CRS, class InvVector, class X, class Y>
    void blockILUBacksolve (const CRS& lower,
                            const CRS& upper,
                            const InvVector& inv,
                            X& v, const Y& d)
    {
      // iterator types
      typedef typename Y :: block_type  dblock;
      typedef typename X :: block_type  vblock;
      typedef typename X :: size_type   size_type ;

      const size_type iEnd = lower.rows();
      const size_type lastRow = iEnd - 1;
      if( iEnd != upper.rows() )
      {
        DUNE_THROW(ISTLError,"ILU::blockILUBacksolve: lower and upper rows must be the same");
      }

      // lower triangular solve
      for( size_type i=0; i<iEnd; ++ i )
      {
        dblock rhsValue( d[ i ] );
        auto&& rhs = Impl::asVector(rhsValue);
        const size_type rowI     = lower.rows_[ i ];
        const size_type rowINext = lower.rows_[ i+1 ];

        for( size_type col = rowI; col < rowINext; ++ col )
          Impl::asMatrix(lower.values_[ col ]).mmv( Impl::asVector(v[ lower.cols_[ col ] ] ), rhs );

        Impl::asVector(v[ i ]) = rhs;  // Lii = I
      }

      // upper triangular solve
      for( size_type i=0; i<iEnd; ++ i )
      {
        auto&& vBlock = Impl::asVector(v[ lastRow - i ]);
        vblock rhsValue ( v[ lastRow - i ] );
        auto&& rhs = Impl::asVector(rhsValue);
        const size_type rowI     = upper.rows_[ i ];
        const size_type rowINext = upper.rows_[ i+1 ];

        for( size_type col = rowI; col < rowINext; ++ col )
          Impl::asMatrix(upper.values_[ col ]).mmv( Impl::asVector(v[ upper.cols_[ col ] ]), rhs );

        // apply inverse and store result
        Impl::asMatrix(inv[ i ]).mv(rhs, vBlock);
      }
    }

  } // end namespace ILU

  /** @} end documentation */

} // end namespace

#endif
