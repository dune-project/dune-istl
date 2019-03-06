// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_TEST_MATRIXTEST_HH
#define DUNE_ISTL_TEST_MATRIXTEST_HH

/** \file
 * \brief Infrastructure for testing the dune-istl matrix interface
 *
 * This file contains various methods that test parts of the dune-istl matrix interface.
 * They only test very general features, i.e., features that every dune-istl matrix should
 * have.
 *
 * At the same time, these tests should help to define what the dune-istl matrix interface
 * actually is.  They may not currently define the entire interface, but they are a lower
 * bound: any feature that is tested here is part of the dune-istl matrix interface.
 */

#include <dune/common/exceptions.hh>
#include <dune/common/test/iteratortest.hh>

namespace Dune
{
  template <typename Matrix>
  void testVectorSpaceOperations(const Matrix& m)
  {
    // Make a mutable copy of the argument
    Matrix mMutable = m;

    // operator+=
    mMutable += m;
    // operator-=
    mMutable -= m;

    // operator*=
    mMutable *= 0.5;

    // operator/=
    mMutable /= 0.5;

    // axpy -- not sure whether that really is an interface method
    //mMutable.axpy(0.5,m);
  }

}  // namespace Dune

#endif
