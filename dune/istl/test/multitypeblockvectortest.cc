// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \file
 * \brief Test the MultiTypeBlockVector data structure
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockvector.hh>

using namespace Dune;

int main(int argc, char** argv) try
{
  MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > > multiVector;

  std::integral_constant<int, 0> _0;
  std::integral_constant<int, 1> _1;

  multiVector[_0] = {{1,0,0},
                     {0,1,0},
                     {0,0,1}};

  multiVector[_1] = {3.14, 42};

  // test operator<<
  std::cout << multiVector << std::endl;

  // test method 'count'
  std::cout << "multi vector has " << multiVector.count() << " first level blocks" << std::endl;

  static_assert(multiVector.size()==2, "Method MultiTypeBlockVector::size() returned wrong value!");

  if (multiVector.count() != 2)
    DUNE_THROW(Exception, "Method MultiTypeBlockVector::count returned wrong value!");

  // Test copy construction
  auto multiVector2 = multiVector;

  // Test assignment operator
  multiVector2 = multiVector;

  // Test operator+=
  multiVector2 += multiVector;

  // Test operator-=
  multiVector2 -= multiVector;

  // Test multiplication with scalar
  multiVector2 *= (double)0.5;
  multiVector2 *= (int)2;
  multiVector2 *= (float)0.5;

  // Test axpy
  multiVector2.axpy(-1, multiVector);

  // Test two_norm
  std::cout << "multivector2 has two_norm: " << multiVector2.two_norm() << std::endl;

  // Test two_norm2
  std::cout << "multivector2 has two_norm2: " << multiVector2.two_norm2() << std::endl;

  // Test operator*
  std::cout << multiVector * multiVector2 << std::endl;

  // Test method 'dot'
  std::cout << multiVector.dot(multiVector2) << std::endl;

  return 0;
}
catch (Dune::Exception& e)
{
  std::cerr << "DUNE reported an exception: " << e << std::endl;
  return 1;
}
catch (std::exception& e)
{
  std::cerr << "C++ reported an exception: " << e.what() << std::endl;
  return 2;
} catch (...)
{
  std::cerr << "Unknown exception encountered!" << std::endl;
  return 3;
}