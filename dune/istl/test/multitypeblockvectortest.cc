// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
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
#include <complex>

#include <dune/common/deprecated.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/float_cmp.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/test/vectortest.hh>

using namespace Dune;

template<typename... Args>
void testMultiVector(const MultiTypeBlockVector<Args...>& multiVector)
{
    // Test whether the vector exports 'size_type', and whether that is an integer
    using size_type = typename MultiTypeBlockVector<Args...>::size_type;
    static_assert(std::numeric_limits<size_type>::is_integer, "size_type is not an integer!");

    // Test whether we can use std::tuple_element
    {
      // Do std::tuple_element and operator[] return the same type?
      using TupleElementType = typename std::tuple_element<0, MultiTypeBlockVector<Args...> >::type;
      using BracketType = decltype(multiVector[Indices::_0]);

      // As the return type of const operator[], BracketType will always
      // be a const reference.  We cannot simply strip the const and the &,
      // because entries of a MultiTypeBlockVector can be references themselves.
      // Therefore, always add const& to the result of std::tuple_element as well.
      constexpr bool sameType = std::is_same_v<const TupleElementType&,BracketType>;
      static_assert(sameType, "std::tuple_element does not provide the type of the 0th MultiTypeBlockVector entry!");
    }

    // test operator<<
    std::cout << multiVector << std::endl;

    // test method 'count'
    std::cout << "multi vector has " << multiVector.N() << " first level blocks" << std::endl;

    static_assert(MultiTypeBlockVector<Args...>::size()==2, "Method MultiTypeBlockVector::size() returned wrong value!");

DUNE_NO_DEPRECATED_BEGIN
    if (multiVector.count() != 2)
      DUNE_THROW(Exception, "Method MultiTypeBlockVector::count returned wrong value!");
DUNE_NO_DEPRECATED_END

    if (multiVector.N() != 2)
      DUNE_THROW(Exception, "Method MultiTypeBlockVector::N returned wrong value!");

    if (multiVector.dim() != 11)
      DUNE_THROW(Exception, "Method MultiTypeBlockVector::dim returned wrong value!");

    // Test copy construction
    auto multiVector2 = multiVector;

    // Test assignment operator
    multiVector2 = multiVector;

    // Test operator+=
    testVectorSpaceOperations(multiVector);

    // Test assignment from scalar
    multiVector2 = (double)0.5;
    multiVector2 = (int)2;
    multiVector2 = (float)0.5;

    // Test the various vector norms
    testNorms(multiVector2);

    // Test operator*
    std::cout << multiVector * multiVector2 << std::endl;

    // Test method 'dot'
    std::cout << multiVector.dot(multiVector2) << std::endl;
}

int main(int argc, char** argv) try
{
  using namespace Indices;

  MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > > multiVector;

  multiVector[_0] = {{1,0,0},
                     {0,1,0},
                     {0,0,1}};

  multiVector[_1] = {3.14, 42};

  testMultiVector(multiVector);

  // create a "shallow" copy
  MultiTypeBlockVector<BlockVector<FieldVector<double,3> >&, BlockVector<FieldVector<double,1> >& >
    multiVectorRef(multiVector[_0], multiVector[_1]);

  multiVectorRef[_0][0][0] = 5.0;

  if (!FloatCmp::eq(multiVectorRef[_0][0][0], multiVector[_0][0][0]))
    DUNE_THROW(Exception, "Modifying an entry of the referencing vector failed!");

  testMultiVector(multiVectorRef);

  // test field_type (std::common_type) for some combinations
  static_assert ( std::is_same_v<typename MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > >::field_type,
                                 double > );
  static_assert ( std::is_same_v<typename MultiTypeBlockVector<BlockVector<FieldVector<float,3> >, BlockVector<FieldVector<float,1> > >::field_type,
                                 float > );
  static_assert ( std::is_same_v<typename MultiTypeBlockVector<BlockVector<FieldVector<float,3> >, BlockVector<FieldVector<double,1> > >::field_type,
                                 double > );
  static_assert ( std::is_same_v<typename MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<std::complex<double>,1> > >::field_type,
                                 std::complex<double> > );

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
