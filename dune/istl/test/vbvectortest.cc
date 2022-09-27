// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/vbvector.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/common/test/iteratortest.hh>

#include <dune/istl/test/vectortest.hh>

using namespace Dune;

int main()
{
  TestSuite suite;

  VariableBlockVector<FieldVector<double,1> > v1;
  VariableBlockVector<FieldVector<double,1> > v2 = v1;
  VariableBlockVector<FieldVector<double,1> > v3(10);
  VariableBlockVector<FieldVector<double,1> > v4(10,4);

  v3.resize(20);
  v4.resize(20,8);

  v3 = v4;

  /*
    v3 is now fully initialized due to the former copy operation with
    the initialized vector v4.
    Calling the create iterator is not allowed, now.
    We have to un-initialize it first:
  */
  std::size_t size = 20;
  v3.resize(size); // this makes v3 unitialized again

  // Set block sizes with CreateIterator:
  for (auto cIt = v3.createbegin(); cIt!=v3.createend(); ++cIt)
    cIt.setblocksize(3);

  v3 = 1.0;

  // Test whether something from <algorithm> can be used to set the block sizes
  // We can't use std::fill() here, as that requires a forward iterator, std::fill_n()
  // is more lenient and settles for an output iterator
  v1.resize(size);
  std::fill_n(v1.createbegin(), size, 10);

  // More formally: test whether the CreateIterator is an output iterator in the stl sense
  v1.resize(5);
  testOutputIterator(v1.createbegin(), 5, 10);

  /* Copy-ing specific blocks with `auto` from a VariableBlockVector is tricky, because
   * the returned object will be a reference:
   */
  auto block0_copy_reference = v3[0];
  block0_copy_reference[0] = 4.2; // change first entry in the copy which has reference semantics. This also changes v3!
  suite.check(v3[0][0] != 1.0, "Show auto x = v3[0] has reference semantics")
    << "Unexpected behaviour: v3[0][0] is " << v3[0][0];

  v3[0][0]=1.0; // reset v3

  // For an actual copy, use the Dune::autoCopy() mechanism
  // This will give a BlockVector with the contents of v3[0].
  auto block0_autoCopy = Dune::autoCopy(v3[0]);
  block0_autoCopy[0] = 4.2;
  suite.check(v3[0][0] == 1.0, "Show that v3 was not modified when copying via autoCopy")
    << "Unexpected behaviour: v3[0][0] is " << v3[0][0];


  // Perform more general vector tests:
  testHomogeneousRandomAccessContainer(v3);
  Dune::testConstructibility<VariableBlockVector<FieldVector<double,1> > >();
  testNorms(v3);
  testVectorSpaceOperations(v3);
  testScalarProduct(v3);

  // Perform tests with a scalar vector entry
  VariableBlockVector<double> v5(10);

  testHomogeneousRandomAccessContainer(v5);
  Dune::testConstructibility<VariableBlockVector<double> >();
  testNorms(v5);
  testVectorSpaceOperations(v5);
  testScalarProduct(v5);

  return suite.exit();
}
