#include "vector.hh"

#include <cmath>

namespace Dune
{
  namespace Mock
  {
    Vector::Vector()
    {}

    Vector::Vector( const std::vector<double>& data)
      : data_(data)
    {}

    Vector& Vector::operator+=( const Vector& y )
    {
      for( std::size_t i = 0; i < data_.size(); ++i )
        data_[i] += y.data_[i];
      return *this;
    }

    Vector& Vector::operator*=( double a )
    {
      for( std::size_t i = 0; i < data_.size(); ++i )
        data_[i] *= a;
      return *this;
    }

    void Vector::axpy( double a, const Vector& y )
    {
      for( std::size_t i = 0; i < data_.size(); ++i )
        data_[i] += a * y.data_[i];
    }

    double Vector::dot( const Vector& y ) const
    {
      double result = 0;
      for( std::size_t i = 0; i < data_.size(); ++i )
        result += data_[i] * y.data_[i];
      return result;
    }

    std::size_t Vector::size() const
    {
      return data_.size();
    }

    double& Vector::operator[](unsigned i)
    {
      return data_[i];
    }

    double Vector::operator[](unsigned i) const
    {
      return data_[i];
    }


    double Vector::two_norm() const
    {
      return sqrt( dot(*this) );
    }
  }
}
