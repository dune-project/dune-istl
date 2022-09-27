// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/float_cmp.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/test/matrixtest.hh>

using namespace Dune;

template <class Matrix, class Vector>
int testBCRSMatrix(int size)
{
  // Set up a test matrix
  Matrix mat;
  setupLaplacian(mat, size);

  // Test vector space operations
  testVectorSpaceOperations(mat);

  // Test the matrix norms
  testNorms(mat);

  // Test whether matrix class has the required constructors
  testMatrixConstructibility<Matrix>();

  // Test the matrix vector products
  Vector domain(mat.M());
  domain = 0;
  Vector range(mat.N());

  testMatrixVectorProducts(mat,domain,range);

  return 0;
}

int main(int argc, char** argv)
{
  // Test scalar matrices and vectors
  int ret = testBCRSMatrix<BCRSMatrix<double>, BlockVector<double> >(10);

  // Test block matrices and vectors with trivial blocks
  ret = testBCRSMatrix<BCRSMatrix<FieldMatrix<double,1,1> >, BlockVector<FieldVector<double,1> > >(10);

  return ret;
}
