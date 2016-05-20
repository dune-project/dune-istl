// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/vbvector.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/test/vectortest.hh>

using namespace Dune;

int main()
{
  VariableBlockVector<FieldVector<double,1> > v1;
  VariableBlockVector<FieldVector<double,1> > v2 = v1;
  VariableBlockVector<FieldVector<double,1> > v3(10);
  VariableBlockVector<FieldVector<double,1> > v4(10,4);

  v3.resize(20);
  v4.resize(20,8);

  v3 = v4;

  for (auto cIt = v3.createbegin(); cIt!=v3.createend(); ++cIt)
    cIt.setblocksize(3);

  v3 = 1.0;

  testHomogeneousRandomAccessContainer(v3);
  Dune::testConstructibility<VariableBlockVector<FieldVector<double,1> > >();
  testNorms(v3);
  testVectorSpaceOperations(v3);
  testScalarProduct(v3);

  return 0;
}
