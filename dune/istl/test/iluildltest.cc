#include <config.h>
#include <iostream>

#include <dune/common/deprecated.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/exceptions.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>

#include <dune/istl/preconditioners.hh>

#include "hilbertmatrix.hh"

template< template< class, class, class, int ... > class _Prec, class K = double, int N = 1 >
void testDecomposition ( int n )
{
  using MatrixBlock = Dune::FieldMatrix< K, N, N >;
  using VectorBlock = Dune::FieldVector< K, N >;

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
    if ( y.two_norm() > 1e-8 )
      DUNE_THROW( Dune::Exception, "Method Prec::apply() returned wrong value!");
  }
}


int main(int argc, char** argv)
try {
  testDecomposition< Dune::SeqILDL, double, 1 >( 4 );
DUNE_NO_DEPRECATED_BEGIN // for deprecated SeqILU0
  testDecomposition< Dune::SeqILU0, double, 1 >( 4 );
DUNE_NO_DEPRECATED_END // for deprecated SeqILU0
  testDecomposition< Dune::SeqILU,  double, 1 >( 4 );

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
