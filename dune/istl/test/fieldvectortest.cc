// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/** \file
 * \brief Test the FieldVector class from dune-common with the unit tests for the dune-istl vector interface
 *
 * This test is exceptional: it is testing a class from dune-common!  This is because FieldVector is supposed
 * to implement the dune-istl vector interface, but it resides in dune-common because most other modules
 * also want to use it.  However, the tests for the dune-istl vector interface reside in dune-istl, and therefore
 * the compliance test for FieldVector needs to be done in dune-istl, too.
 */
#include <complex>

#include <dune/common/fvector.hh>

#include <dune/istl/test/vectortest.hh>

using namespace Dune;

int main() try
{
  // Test a double vector
  FieldVector<double,3> vDouble = {1.0, 2.0, 3.0};
  testHomogeneousRandomAccessContainer(vDouble);
  testConstructibility<decltype(vDouble)>();
  testNorms(vDouble);
  testVectorSpaceOperations(vDouble);

  // Test a double vector of length 1
  FieldVector<double,1> vDouble1 = {1.0};
  testHomogeneousRandomAccessContainer(vDouble1);
  testConstructibility<decltype(vDouble1)>();
  testNorms(vDouble1);
  testVectorSpaceOperations(vDouble1);

  // Test a complex vector
  FieldVector<std::complex<double>,3> vComplex = {{1.0, 1.0}, {2.0,2.0}, {3.0,3.0}};
  testHomogeneousRandomAccessContainer(vComplex);
  testConstructibility<decltype(vComplex)>();
  testNorms(vComplex);
  testVectorSpaceOperations(vComplex);

  // Test a complex vector of length 1
  FieldVector<std::complex<double>,1> vComplex1 = {{1.0,3.14}};
  testHomogeneousRandomAccessContainer(vComplex1);
  testConstructibility<decltype(vComplex1)>();
  testNorms(vComplex1);
  testVectorSpaceOperations(vComplex1);

  return 0;
}
catch (std::exception& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
