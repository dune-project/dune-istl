// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef HILBERTMATRIX_HH
#define HILBERTMATRIX_HH

#include <dune/common/fmatrix.hh>

#include <dune/istl/bcrsmatrix.hh>


template< class B >
void setupSP (Dune::BCRSMatrix< B >& A, int n )
{
  using matrix_type = Dune::BCRSMatrix< B >;

  A.setSize( n, n, n*n );
  A.setBuildMode( matrix_type::row_wise );

  for( auto i = A.createbegin(); i != A.createend(); ++i )
    for( typename matrix_type::size_type j = 0; j < A.N(); ++j )
      i.insert( j );
}

template < class B >
void setupHilbertMatrix ( Dune::BCRSMatrix< B >& A, B block, int n  )
{
  using matrix_type = Dune::BCRSMatrix< B >;
  using field_type  = typename matrix_type::field_type;
  using size_type   = typename matrix_type::size_type;

  setupSP( A, n );

  for ( size_type i = 0; i < A.N(); ++i )
    for ( size_type j = 0; j < A.M(); ++j )
    {
      A[ i ][ j ] = block;
      A[ i ][ j ] /= static_cast<field_type>( i + j + 1 );
    }
}

template< class T >
void setupHilbertMatrix ( Dune::BCRSMatrix< Dune::FieldMatrix< T, 1, 1 > >& A, int n  )
{
  setupHilbertMatrix( A, Dune::FieldMatrix< T, 1, 1 >{ 1.0 }, n );
}

template< class T >
void setupHilbertMatrix ( Dune::BCRSMatrix< T >& A, int n  )
{
  setupHilbertMatrix( A, T{ 1.0 }, n );
}

#endif // #ifndef HILBERTMATRIX_HH
