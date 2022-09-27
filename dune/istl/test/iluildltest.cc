// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/exceptions.hh>

#include <dune/common/simd/loop.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>

#include <dune/istl/preconditioners.hh>

#include "hilbertmatrix.hh"


template< template< class, class, class, int ... > class _Prec, class MatrixBlock, class VectorBlock >
void testDecomposition ( int n )
{
  using BlockMatrix = Dune::BCRSMatrix< MatrixBlock >;
  using BlockVector = Dune::BlockVector< VectorBlock >;

  using Prec = _Prec< BlockMatrix, BlockVector, BlockVector >;

  BlockMatrix A;
  setupHilbertMatrix( A, n );

  Prec prec( A, 1.0 );

  for ( int i = 0; i < n; ++i )
  {
    BlockVector x( n ), y( n ), b( n );
    y = 0.0;
    y[ i ] = 1.0;
    A.mv( y, b );
    prec.pre( x, b );
    prec.apply( x, b );
    prec.post( x );

    y -= x;
    if ( Dune::Simd::anyTrue(y.two_norm() > 1e-8) )
      DUNE_THROW( Dune::Exception, "Method Prec::apply() returned wrong value!");
  }
}


int main(int argc, char** argv)
try {

  testDecomposition< Dune::SeqILDL, double, double >( 4 );
  testDecomposition< Dune::SeqILU, double, double >( 4 );
  testDecomposition< Dune::SeqILDL, Dune::FieldMatrix<double,1,1>, Dune::FieldVector<double,1> >( 4 );
  testDecomposition< Dune::SeqILU, Dune::FieldMatrix<double,1,1>, Dune::FieldVector<double,1> >( 4 );
  testDecomposition<Dune::SeqILU, Dune::LoopSIMD<double, 4>, Dune::LoopSIMD<double, 4>>( 4 );

  return 0;
}
catch(Dune::Exception &e)
{
  std::cerr << "Dune reported error: " << e << std::endl;
}
catch (...)
{
  std::cerr << "Unknown exception" << std::endl;
}
