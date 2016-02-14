#include "linearOperator_2d.hh"

namespace Dune
{
  namespace Mock
  {
    LinearOperator_2d::LinearOperator_2d()
    {
      data_.push_back( { 4, 1 } );
      data_.push_back( { 1, 3 } );
    }

    void LinearOperator_2d::apply( const Vector& x, Vector& y ) const
    {
      for( size_t i = 0; i < y.data_.size(); ++i )
      {
        y.data_[i] = 0;
        for( size_t j = 0; j < x.data_.size(); ++j )
          y.data_[i] += data_[i][j] * x.data_[j];
      }
    }

    void LinearOperator_2d::applyscaleadd( double a, const Vector& x, Vector& y ) const
    {
      for( size_t i = 0; i < y.data_.size(); ++i )
        for( size_t j = 0; j < x.data_.size(); ++j )
          y.data_[i] += a * data_[i][j] * x.data_[j];
    }
  }
}
