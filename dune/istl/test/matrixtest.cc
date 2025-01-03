// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/** \file
    \brief Unit tests for the different dynamic matrices provided by ISTL
 */

#include <fenv.h>

#include <dune/common/fmatrix.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/istl/blocklevel.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/istl/btdmatrix.hh>
#include <dune/istl/scaledidmatrix.hh>
#include <dune/istl/test/matrixtest.hh>


using namespace Dune;

int main()
{


#if defined( __APPLE__ ) or defined( __MINGW32__ ) or defined(_MSC_VER)
  feraiseexcept(FE_INVALID);
#else
  feenableexcept(FE_INVALID);
#endif


  // ////////////////////////////////////////////////////////////
  //   Test the Matrix class -- a scalar dense dynamic matrix
  // ////////////////////////////////////////////////////////////

  {
    Matrix<double> matrixScalar(10,10);
    for (int i=0; i<10; i++)
      for (int j=0; j<10; j++)
        matrixScalar[i][j] = (i+j)/((double)(i*j+1));        // just anything

    BlockVector<double> x(10), y(10);
    testMatrix(matrixScalar, x, y);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the Matrix class -- a block-valued dense dynamic matrix
  // ////////////////////////////////////////////////////////////

  Matrix<FieldMatrix<double,3,3> > matrix(10,10);
  for (int i=0; i<10; i++)
    for (int j=0; j<10; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          matrix[i][j][k][l] = (i+j)/((double)(k*l+1));            // just anything

  testSuperMatrix(matrix);

  Matrix<double> nonquadraticMatrix(1,2);
  {
    size_t n = 1;
    for (size_t i=0; i<1; i++)
      for (size_t j=0; j<2; j++)
        nonquadraticMatrix[i][j] = n++;
  }

  testMatrixTranspose(nonquadraticMatrix);


  // BCRSMatrix tests have been moved to bcrsmatrixtest.cc

  ///////////////////////////////////////////////////////////////////////////
  //   Test the BDMatrix class -- a dynamic block-diagonal matrix
  ///////////////////////////////////////////////////////////////////////////

  {
    BDMatrix<double> bdMatrix(3);
    bdMatrix = 4.0;

    BlockVector<double> x(3), y(3);
    testMatrix(bdMatrix, x, y);

    // Test construction from initializer list
    BDMatrix<double> bdMatrix2 = {1.0, 2.0, 3.0};
    testMatrix(bdMatrix2, x, y);

    // Test equation-solving
    BDMatrix<double> bdMatrix3 = {1.0, 2.0, 3.0};
    testMatrixSolve<BDMatrix<double>, BlockVector<double> >(bdMatrix3);

    // Test whether inversion works
    bdMatrix3.invert();

    // test whether resizing works
    bdMatrix2.setSize(5);
    bdMatrix2 = 4.0;
    x.resize(5);
    y.resize(5);
    testMatrix(bdMatrix2, x, y);

  }

  // ////////////////////////////////////////////////////////////////////////
  //   Test the BDMatrix class with FieldMatrix entries
  // ////////////////////////////////////////////////////////////////////////

  BDMatrix<FieldMatrix<double,4,4> > bdMatrix(2);
  bdMatrix = 4.0;

  testSuperMatrix(bdMatrix);

  // Test construction from initializer list
  BDMatrix<FieldMatrix<double,2,2> > bdMatrix2 = { {{1,0},{0,1}}, {{0,1},{-1,0}}};

  // Test equation-solving
  testMatrixSolve<BDMatrix<FieldMatrix<double,2,2> >, BlockVector<FieldVector<double,2> > >(bdMatrix2);

  // Test whether inversion works
  bdMatrix2.invert();

  // Run matrix tests on this matrix
  testSuperMatrix(bdMatrix2);

  // test whether resizing works
  bdMatrix2.setSize(5);
  bdMatrix2 = 4.0;
  testSuperMatrix(bdMatrix2);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the BTDMatrix class -- a dynamic block-tridiagonal matrix
  //   a) the scalar case
  // ////////////////////////////////////////////////////////////////////////

  {
    BTDMatrix<double> btdMatrixScalar(4);
    using size_type = BTDMatrix<double>::size_type;

    btdMatrixScalar = 4.0;

    BlockVector<double> x(4), y(4);
    testMatrix(btdMatrixScalar, x, y);

    btdMatrixScalar = 0.0;
    for (size_type i=0; i<btdMatrixScalar.N(); i++)    // diagonal
      btdMatrixScalar[i][i] = 1+i;

    for (size_type i=0; i<btdMatrixScalar.N()-1; i++)
      btdMatrixScalar[i][i+1] = 2+i;               // first off-diagonal

    testMatrixSolve<BTDMatrix<double>, BlockVector<double> >(btdMatrixScalar);

    // test a 1x1 BTDMatrix, because that is a special case
    BTDMatrix<double> btdMatrixScalar_1x1(1);
    btdMatrixScalar_1x1 = 1.0;
    x.resize(1);
    y.resize(1);
    testMatrix(btdMatrixScalar_1x1, x, y);

    // test whether resizing works
    btdMatrixScalar_1x1.setSize(5);
    btdMatrixScalar_1x1 = 4.0;
    x.resize(5);
    y.resize(5);
    testMatrix(btdMatrixScalar_1x1, x, y);
  }

  ///////////////////////////////////////////////////////////////////////////
  //   Test the BTDMatrix class -- a dynamic block-tridiagonal matrix
  //   b) the scalar case with FieldMatrix entries
  ///////////////////////////////////////////////////////////////////////////

  BTDMatrix<FieldMatrix<double,1,1> > btdMatrixScalar(4);
  typedef BTDMatrix<FieldMatrix<double,1,1> >::size_type size_type;

  btdMatrixScalar = 4.0;

  testSuperMatrix(btdMatrixScalar);

  btdMatrixScalar = 0.0;
  for (size_type i=0; i<btdMatrixScalar.N(); i++)    // diagonal
    btdMatrixScalar[i][i] = 1+i;

  for (size_type i=0; i<btdMatrixScalar.N()-1; i++)
    btdMatrixScalar[i][i+1] = 2+i;               // first off-diagonal

  testMatrixSolve<BTDMatrix<FieldMatrix<double,1,1> >, BlockVector<FieldVector<double,1> > >(btdMatrixScalar);

  // test a 1x1 BTDMatrix, because that is a special case
  BTDMatrix<FieldMatrix<double,1,1> > btdMatrixScalar_1x1(1);
  btdMatrixScalar_1x1 = 1.0;
  testSuperMatrix(btdMatrixScalar_1x1);

  // test whether resizing works
  btdMatrixScalar_1x1.setSize(5);
  btdMatrixScalar_1x1 = 4.0;
  testSuperMatrix(btdMatrixScalar_1x1);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the BTDMatrix class -- a dynamic block-tridiagonal matrix
  //   c) the block-valued case
  // ////////////////////////////////////////////////////////////////////////

  BTDMatrix<FieldMatrix<double,2,2> > btdMatrix(4);
  typedef BTDMatrix<FieldMatrix<double,2,2> >::size_type size_type;

  btdMatrix = 0.0;
  for (size_type i=0; i<btdMatrix.N(); i++)    // diagonal
    btdMatrix[i][i] = ScaledIdentityMatrix<double,2>(1+i);

  for (size_type i=0; i<btdMatrix.N()-1; i++)
    btdMatrix[i][i+1] = ScaledIdentityMatrix<double,2>(2+i);               // upper off-diagonal
  for (size_type i=1; i<btdMatrix.N(); i++)
    btdMatrix[i-1][i] = ScaledIdentityMatrix<double,2>(2+i);               // lower off-diagonal

  // add some off diagonal stuff to the blocks in the matrix
  // diagonals
  btdMatrix[0][0][0][1] = 2;
  btdMatrix[0][0][1][0] = -1;

  btdMatrix[1][1][0][1] = 2;
  btdMatrix[1][1][1][0] = 3;

  btdMatrix[2][2][0][1] = 2;
  btdMatrix[2][2][0][0] += sqrt(2.);
  btdMatrix[2][2][1][0] = 3;

  btdMatrix[3][3][0][1] = -1;
  btdMatrix[3][3][0][0] -= 0.5;
  btdMatrix[3][3][1][0] = 2;

  // off diagonals
  btdMatrix[0][1][0][1] = std::sqrt(2);
  btdMatrix[1][0][0][1] = std::sqrt(2);

  btdMatrix[1][0][1][0] = -13./17.;
  btdMatrix[1][2][0][1] = -1./std::sqrt(2);
  btdMatrix[1][2][1][0] = -13./17.;

  btdMatrix[2][1][0][1] = -13./17.;
  btdMatrix[2][1][1][0] = -13./17.;
  btdMatrix[2][3][0][1] = -1./std::sqrt(2);
  btdMatrix[2][3][1][0] = -17.;

  btdMatrix[3][2][0][1] = 1.;
  btdMatrix[3][2][1][0] = 1.;


  BTDMatrix<FieldMatrix<double,2,2> > btdMatrixThrowAway = btdMatrix;    // the test method overwrites the matrix
  testSuperMatrix(btdMatrixThrowAway);

  testMatrixSolve<BTDMatrix<FieldMatrix<double,2,2> >, BlockVector<FieldVector<double,2> > >(btdMatrix);

  // test a 1x1 BTDMatrix, because that is a special case
  BTDMatrix<FieldMatrix<double,2,2> > btdMatrix_1x1(1);
  btdMatrix_1x1 = 1.0;
  testSuperMatrix(btdMatrix_1x1);

  // test whether resizing works
  btdMatrix_1x1.setSize(5);
  btdMatrix_1x1 = 4.0;
  testSuperMatrix(btdMatrix_1x1);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the FieldMatrix class
  // ////////////////////////////////////////////////////////////////////////
  typedef FieldMatrix<double,4,4>::size_type size_type;
  FieldMatrix<double,4,4> fMatrix;

  for (size_type i=0; i<fMatrix.N(); i++)
    for (size_type j=0; j<fMatrix.M(); j++)
      fMatrix[i][j] = (i+j)/3;        // just anything
  FieldVector<double,4> fvX;
  FieldVector<double,4> fvY;

  testMatrix(fMatrix, fvX, fvY);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the 1x1 specialization of the FieldMatrix class
  // ////////////////////////////////////////////////////////////////////////

  FieldMatrix<double,1,1> fMatrix1x1;
  fMatrix1x1[0][0] = 2.3;    // just anything

  FieldVector<double,1> fvX1;
  FieldVector<double,1> fvY1;

  testMatrix(fMatrix1x1, fvX1, fvY1);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the DiagonalMatrix class
  // ////////////////////////////////////////////////////////////////////////

  DiagonalMatrix<double,4> dMatrix1;
  dMatrix1 = 3.1459;
  testMatrix(dMatrix1, fvX, fvY);

  DiagonalMatrix<double,4> dMatrix2(3.1459);
  testMatrix(dMatrix2, fvX, fvY);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the ScaledIdentityMatrix class
  // ////////////////////////////////////////////////////////////////////////

  ScaledIdentityMatrix<double,4> sIdMatrix;
  sIdMatrix = 3.1459;

  testMatrix(sIdMatrix, fvX, fvY);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the Matrix class with complex scalars
  // ////////////////////////////////////////////////////////////////////////

  {
    Matrix<std::complex<double>> matrixComplex(10,10);
    for (int i=0; i<10; i++)
      for (int j=0; j<10; j++)
        matrixComplex[i][j] = std::complex<double>((i+j)/((double)(i*j+1)), (i-j)/((double)(i*j+1)));

    BlockVector<std::complex<double>> cx(10), cy(10);
    testMatrix(matrixComplex, cx, cy);
  }

  // ////////////////////////////////////////////////////////////////////////
  //   Test the FieldMatrix class with complex scalars
  // ////////////////////////////////////////////////////////////////////////

  {
    FieldMatrix<std::complex<double>,4,4> cfMatrix;
    for (size_type i=0; i<cfMatrix.N(); i++)
      for (size_type j=0; j<cfMatrix.M(); j++)
        cfMatrix[i][j] = std::complex<double>((i+j)/3.0, (i-j)/3.0);
    FieldVector<std::complex<double>,4> cfvX;
    FieldVector<std::complex<double>,4> cfvY;

    testMatrix(cfMatrix, cfvX, cfvY);
  }
}
