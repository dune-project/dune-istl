#ifndef DUNE_ISTL_ILDL_HH
#define DUNE_ISTL_ILDL_HH

#include "ilu.hh"

/**
 * \file
 *
 * \brief   Incomplete LDL decomposition
 * \author  Martin Nolte
 **/

namespace Dune
{

  // bildl_subtractBCT
  // -----------------

  template< class K, int m, int n >
  inline static void bildl_subtractBCT ( const FieldMatrix< K, m, n > &B, const FieldMatrix< K, m, n > &CT, FieldMatrix< K, m, n > &A )
  {
    for( int i = 0; i < m; ++i )
    {
      for( int j = 0; j < n; ++j )
      {
        for( int k = 0; k < n; ++k )
          A[ i ][ j ] -= B[ i ][ k ] * CT[ j ][ k ];
      }
    }
  }

  template< class Matrix >
  inline static void bildl_subtractBCT ( const Matrix &B, const Matrix &CT, Matrix &A )
  {
    for( auto i = A.begin(), iend = A.end(); i = iend; ++i )
    {
      auto &&A_i = *i;
      auto &&B_i = B[ i.index() ];
      const auto ikend = B_i.end();
      for( auto j = A_i.begin(), jend = A_i.end(); j != jend; ++j )
      {
        auto &&A_ij = *j;
        auto &&CT_j = CT[ j.index() ];
        const auto jkend = CT_j.end();
        for( auto ik = B_i.begin(), jk = CT_j.begin(); (ik != ikend) && (jk != jkend); )
        {
          if( ik.index() == jk.index() )
          {
            bildl_subtractBCT( *ik, *jk, A_ij );
            ++ik; ++jk;
          }
          else if( ik.index() < jk.index() )
            ++ik;
          else
            ++jk;
        }
      }
    }
  }



  // bildl_decompose
  // ---------------

  /**
   * \brief  compute ILDL decomposition of a symmetric matrix A
   * \author Martin Nolte
   *
   * \param[inout]  A  matrix to decompose
   *
   * \note A is overwritten by the factorization.
   * \note Only the lower half of A is used.
   **/
  template< class Matrix >
  inline void bildl_decompose ( Matrix &A )
  {
    for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
    {
      auto &&A_i = *i;

      auto ij = A_i.begin();
      for( ; ij.index() < i.index(); ++ij )
      {
        auto &&A_ij = *ij;
        auto &&A_j = A[ ij.index() ];

        // store L_ij Dj in A_ij (note: for k < i: A_kj = L_kj)
        // L_ij Dj = A_ij - \sum_{k < j} (L_ik D_k) L_jk^T
        auto ik = A_i.begin();
        auto jk = A_j.begin();
        while( (ik != ij) && (jk.index() < ij.index()) )
        {
          if( ik.index() == jk.index() )
          {
            bildl_subtractBCT(*ik, *jk, A_ij);
            ++ik; ++jk;
          }
          else if( ik.index() < jk.index() )
            ++ik;
          else
            ++jk;
        }
      }

      if( ij.index() != i.index() )
        DUNE_THROW( ISTLError, "diagonal entry missing" );

      // update diagonal and multiply A_ij by D_j^{-1}
      auto &&A_ii = *ij;
      for( auto ik = A_i.begin(); ik != ij; ++ik )
      {
        auto &&A_ik = *ik;
        const auto &A_k = A[ ik.index() ];

        auto B = A_ik;
        A_ik.rightmultiply( *A_k.find( ik.index() ) );
        bildl_subtractBCT( B, A_ik, A_ii );
      }
      try
      {
        A_ii.invert();
      }
      catch( const Dune::FMatrixError &e )
      {
        DUNE_THROW( MatrixBlockError, "ILDL failed to invert matrix block A[" << i.index() << "][" << ij.index() << "]" << e.what(); th__ex.r = i.index(); th__ex.c = ij.index() );
      }
    }
  }



  // bildl_backsolve
  // ---------------

  template< class Matrix, class X, class Y >
  inline void bildl_backsolve ( const Matrix &A, X &v, const Y &d, bool isLowerTriangular = false )
  {
    // solve L v = d, note: Lii = I
    for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
    {
      const auto &A_i = *i;
      v[ i.index() ] = d[ i.index() ];
      for( auto ij = A_i.begin(); ij.index() < i.index(); ++ij )
        (*ij).mmv( v[ ij.index() ], v[ i.index() ] );
    }

    // solve D w = v, note: diagonal stores Dii^{-1}
    if( isLowerTriangular )
    {
      // The matrix is lower triangular, so the diagonal entry is the
      // last one in each row.
      for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
      {
        const auto &A_i = *i;
        const auto ii = A_i.beforeEnd();
        assert( ii.index() == i.index() );
        auto rhs = v[ i.index() ];
        ii->mv( rhs, v[ i.index() ] );
      }
    }
    else
    {
      // Without assumptions on the sparsity pattern we have to search
      // for the diagonal entry in each row.
      for( auto i = A.begin(), iend = A.end(); i != iend; ++i )
      {
        const auto &A_i = *i;
        const auto ii = A_i.find( i.index() );
        assert( ii.index() == i.index() );
        auto rhs = v[ i.index() ];
        ii->mv( rhs, v[ i.index() ] );
      }
    }

    // solve L^T v = w, note: only L is stored
    // note: we perform the operation column-wise from right to left
    for( auto i = A.beforeEnd(), iend = A.beforeBegin(); i != iend; --i )
    {
      const auto &A_i = *i;
      for( auto ij = A_i.begin(); ij.index() < i.index(); ++ij )
        (*ij).mmtv( v[ i.index() ], v[ ij.index() ] );
    }
  }

} // namespace Dune

#endif // #ifndef DUNE_ISTL_ILDL_HH
