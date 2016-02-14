#ifndef DUNE_ISTL_TESTS_MOCK_LINEAR_OPERATOR_HH
#define DUNE_ISTL_TESTS_MOCK_LINEAR_OPERATOR_HH

#include <vector>

#include "dune/istl/operators.hh"
#include "dune/istl/preconditioner.hh"

#include "vector.hh"

namespace Dune
{
  namespace Mock
  {
//    template <class X>
//    class LinearOperator : public Dune::LinearOperator<X,X>
//    {
//      void apply(const X& x, X& y)
//      {
//        using Index = decltype(x.rows());

//        if( x.rows() == 1 )
//        {
//          y = x;
//          return;
//        }

//        y[0] = 2*x[0] + x[1];
//        for( Index i=1; i<x.rows()-1; ++i )
//          y[i] = x[i-1] + 4*x[i] + x[i+1];
//      }
//    };

    class MassMatrix : public Dune::LinearOperator<Vector,Vector>
    {
      void apply(const Vector& x, Vector& y) const
      {
        y *= 0;
        applyscaleadd( 1, x, y );
      }

      void applyscaleadd( double a , const Vector& x, Vector& y ) const
      {
        if( x.size() == 1 )
        {
          auto z = x;
          y += ( z*= a );
          return;
        }

        y[0] += a*x[0];
        y[x.size()-1] += a*x[x.size()-1];
        for( std::size_t i=1; i < x.size() - 1; ++i )
          y[i] += a/h * ( x[i-1] + 2*x[i] + x[i+1] );

      }

      double h = 1e-2;
    };

    class MassJacobi : public Dune::Preconditioner<Vector,Vector>
    {
      void pre( Vector&, Vector& )
      {}

      void post( Vector& )
      {}

      void apply( Vector& y, const Vector& x )
      {
        if( x.size() == 1 )
        {
          y = x;
          return;
        }

        y[0] = x[0];
        y[x.size()-1] = x[x.size()-1];
        for( std::size_t i=1; i<x.size()-1; ++i )
          y[i] = 2 * h * x[i];
      }

      double h = 1e-2;
    };
  }
}

#endif // DUNE_ISTL_TESTS_MOCK_LINEAR_OPERATOR_HH
