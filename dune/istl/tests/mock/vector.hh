#ifndef DUNE_ISTL_TESTS_MOCK_VECTOR_HH
#define DUNE_ISTL_TESTS_MOCK_VECTOR_HH

#include <vector>

#include <dune/common/typetraits.hh>

namespace Dune
{
  namespace Mock
  {
    struct Vector
    {
      using field_type = double;

      Vector();

      Vector( const std::vector<double>& data );

      Vector& operator+=( const Vector& y );

      Vector& operator*=( double a );

      void axpy( double a, const Vector& y );

      double dot(const Vector& y) const;

      std::size_t size() const;

      double& operator[](unsigned i);

      double operator[](unsigned i) const;

      double two_norm() const;

      std::vector<double> data_;
    };
  }

  template <>
  struct FieldTraits<Mock::Vector>
  {
    using field_type = double;
    using real_type = double;
  };
}

#endif // DUNE_ISTL_TESTS_MOCK_VECTOR_HH
