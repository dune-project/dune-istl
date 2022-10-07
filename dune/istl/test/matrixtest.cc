// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/** \file
    \brief Unit tests for the different dynamic matrices provided by ISTL
 */
#include "config.h"

#include <fenv.h>

#include <dune/common/fmatrix.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/istl/blocklevel.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/istl/btdmatrix.hh>
#include <dune/istl/scaledidmatrix.hh>


using namespace Dune;

// forward decls

template <class MatrixType>
void testSuperMatrix(MatrixType& matrix);
template <class MatrixType, class X, class Y>
void testMatrix(MatrixType& matrix, X& x, Y& y);
template <class MatrixType, class VectorType>
void testSolve(const MatrixType& matrix);

template <class MatrixType>
void testSuperMatrix(MatrixType& matrix)
{
  // ////////////////////////////////////////////////////////
  //   Check the types which are exported by the matrix
  // ////////////////////////////////////////////////////////

  typedef typename MatrixType::field_type field_type;

  typedef typename MatrixType::block_type block_type;

  typedef typename MatrixType::size_type size_type;

  using allocator_type [[maybe_unused]] = typename MatrixType::allocator_type;

  size_type n = matrix.N();
  size_type m = matrix.M();
  BlockVector<FieldVector<field_type, block_type::cols> > x(m);
  BlockVector<FieldVector<field_type, block_type::rows> > y(n);

  testMatrix(matrix, x, y);
}


template <class MatrixType, class X, class Y>
void testMatrix(MatrixType& matrix, X& x, Y& y)
{
  // ////////////////////////////////////////////////////////
  //   Check the types which are exported by the matrix
  // ////////////////////////////////////////////////////////

  typedef typename MatrixType::field_type field_type;

  typedef typename FieldTraits<field_type>::real_type real_type;

  using block_type [[maybe_unused]] = typename MatrixType::block_type;

  using row_type [[maybe_unused]] = typename MatrixType::row_type;

  typedef typename MatrixType::size_type size_type;

  using RowIterator [[maybe_unused]] = typename MatrixType::RowIterator;

  using ConstRowIterator [[maybe_unused]] = typename MatrixType::ConstRowIterator;

  using ColIterator [[maybe_unused]] = typename MatrixType::ColIterator;

  using ConstColIterator [[maybe_unused]] = typename MatrixType::ConstColIterator;

  static_assert(maxBlockLevel<MatrixType>() >= 0, "Block level has to be at least 1 for a matrix!");

  // ////////////////////////////////////////////////////////
  //   Count number of rows, columns, and nonzero entries
  // ////////////////////////////////////////////////////////

  typename MatrixType::RowIterator rowIt    = matrix.begin();
  typename MatrixType::RowIterator rowEndIt = matrix.end();

  typename MatrixType::size_type numRows = 0, numEntries = 0;

  for (; rowIt!=rowEndIt; ++rowIt) {

    typename MatrixType::ColIterator colIt    = rowIt->begin();
    typename MatrixType::ColIterator colEndIt = rowIt->end();

    for (; colIt!=colEndIt; ++colIt) {
      assert(matrix.exists(rowIt.index(), colIt.index()));
      numEntries++;
    }

    numRows++;

  }

  assert (numRows == matrix.N());

  // ///////////////////////////////////////////////////////////////
  //   Count number of rows, columns, and nonzero entries again.
  //   This time use the const iterators
  // ///////////////////////////////////////////////////////////////

  typename MatrixType::ConstRowIterator constRowIt    = matrix.begin();
  typename MatrixType::ConstRowIterator constRowEndIt = matrix.end();

  numRows    = 0;
  numEntries = 0;

  for (; constRowIt!=constRowEndIt; ++constRowIt) {

    typename MatrixType::ConstColIterator constColIt    = constRowIt->begin();
    typename MatrixType::ConstColIterator constColEndIt = constRowIt->end();

    for (; constColIt!=constColEndIt; ++constColIt)
      numEntries++;

    numRows++;

  }

  assert (numRows == matrix.N());

  // ////////////////////////////////////////////////////////
  //   Count number of rows, columns, and nonzero entries
  //   This time we're counting backwards
  // ////////////////////////////////////////////////////////

  rowIt    = matrix.beforeEnd();
  rowEndIt = matrix.beforeBegin();

  numRows    = 0;
  numEntries = 0;

  for (; rowIt!=rowEndIt; --rowIt) {

    typename MatrixType::ColIterator colIt    = rowIt->beforeEnd();
    typename MatrixType::ColIterator colEndIt = rowIt->beforeBegin();

    for (; colIt!=colEndIt; --colIt) {
      assert(matrix.exists(rowIt.index(), colIt.index()));
      numEntries++;
    }

    numRows++;

  }

  assert (numRows == matrix.N());

  // ///////////////////////////////////////////////////////////////
  //   Count number of rows, columns, and nonzero entries again.
  //   This time use the const iterators and count backwards.
  // ///////////////////////////////////////////////////////////////

  constRowIt    = matrix.beforeEnd();
  constRowEndIt = matrix.beforeBegin();

  numRows    = 0;
  numEntries = 0;

  for (; constRowIt!=constRowEndIt; --constRowIt) {

    typename MatrixType::ConstColIterator constColIt    = constRowIt->beforeEnd();
    typename MatrixType::ConstColIterator constColEndIt = constRowIt->beforeBegin();

    for (; constColIt!=constColEndIt; --constColIt)
      numEntries++;

    numRows++;

  }

  assert (numRows == matrix.N());

  // ///////////////////////////////////////////////////////
  //   More dimension stuff
  // ///////////////////////////////////////////////////////

  size_type n = matrix.N(); ++n;
  size_type m = matrix.M(); ++m;

  // ///////////////////////////////////////////////////////
  //   Test assignment operators and the copy constructor
  // ///////////////////////////////////////////////////////

  // assignment from other matrix
  MatrixType secondMatrix;
  secondMatrix = matrix;

  // assignment from scalar
  matrix = 0;

  // The copy constructor
  [[maybe_unused]] MatrixType thirdMatrix(matrix);

  // ///////////////////////////////////////////////////////
  //   Test component-wise operations
  // ///////////////////////////////////////////////////////

  matrix *= real_type(1.0);
  matrix /= real_type(1.0);

  matrix += secondMatrix;
  matrix -= secondMatrix;

  // ///////////////////////////////////////////////////////////
  //   Test the various matrix-vector multiplications
  // ///////////////////////////////////////////////////////////

  Y yy=y;

  matrix.mv(x,yy);

  matrix.mtv(x,yy);

  matrix.umv(x,y);

  matrix.umtv(x,y);

  matrix.umhv(x,y);

  matrix.mmv(x,y);

  matrix.mmtv(x,y);

  matrix.mmhv(x,y);

  matrix.usmv(field_type(1.0),x,y);

  matrix.usmtv(field_type(1.0),x,y);

  matrix.usmhv(field_type(1.0),x,y);

  // //////////////////////////////////////////////////////////////
  //   Test the matrix norms
  // //////////////////////////////////////////////////////////////

  real_type frobenius_norm = matrix.frobenius_norm();

  frobenius_norm += matrix.frobenius_norm2();

  frobenius_norm += matrix.infinity_norm();

  frobenius_norm += matrix.infinity_norm_real();

}

// ///////////////////////////////////////////////////////////////////
//   Test the solve()-method for those matrix classes that have it
// ///////////////////////////////////////////////////////////////////
template <class MatrixType, class VectorType>
void testSolve(const MatrixType& matrix)
{
  typedef typename VectorType::size_type size_type;

  // create some right hand side
  VectorType b(matrix.N());
  for (size_type i=0; i<b.size(); i++)
    b[i] = i;

  // solution vector
  VectorType x(matrix.M());

  // Solve the system
  matrix.solve(x,b);

  // compute residual
  matrix.mmv(x,b);

  if (b.two_norm() > 1e-10)
    DUNE_THROW(ISTLError, "Solve() method doesn't appear to produce the solution!");
}

// //////////////////////////////////////////////////////////////
//   Test transposing the matrix
// //////////////////////////////////////////////////////////////
template <class MatrixType>
void testTranspose(const MatrixType& matrix)
{
  MatrixType transposedMatrix = matrix.transpose();

  for(size_t i = 0; i < matrix.N(); i++)
    for(size_t j = 0; j < matrix.M(); j++)
      if(fabs(transposedMatrix[j][i] - matrix[i][j]) > 1e-10)
        DUNE_THROW(ISTLError, "transpose() method produces wrong result!");
}

int main()
{

  // feenableexcept does not exist on OS X or windows
#if not defined( __APPLE__ ) and not defined( __MINGW32__ )
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

  Matrix<FieldMatrix<double,1,1> > nonquadraticMatrix(1,2);
  {
    size_t n = 1;
    for (size_t i=0; i<1; i++)
      for (size_t j=0; j<2; j++)
        nonquadraticMatrix[i][j] = n++;
  }

  testTranspose(nonquadraticMatrix);


  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class -- a sparse dynamic matrix
  // ////////////////////////////////////////////////////////////

  {
    BCRSMatrix<double> bcrsMatrix(4,4, BCRSMatrix<double>::random);

    bcrsMatrix.setrowsize(0,2);
    bcrsMatrix.setrowsize(1,3);
    bcrsMatrix.setrowsize(2,3);
    bcrsMatrix.setrowsize(3,2);

    bcrsMatrix.endrowsizes();

    bcrsMatrix.addindex(0, 0);
    bcrsMatrix.addindex(0, 1);

    bcrsMatrix.addindex(1, 0);
    bcrsMatrix.addindex(1, 1);
    bcrsMatrix.addindex(1, 2);

    bcrsMatrix.addindex(2, 1);
    bcrsMatrix.addindex(2, 2);
    bcrsMatrix.addindex(2, 3);

    bcrsMatrix.addindex(3, 2);
    bcrsMatrix.addindex(3, 3);

    bcrsMatrix.endindices();

    typedef BCRSMatrix<double>::RowIterator RowIterator;
    typedef BCRSMatrix<double>::ColIterator ColIterator;

    for(RowIterator row = bcrsMatrix.begin(); row != bcrsMatrix.end(); ++row)
      for(ColIterator col = row->begin(); col != row->end(); ++col)
        *col = 1.0 + (double) row.index() * (double) col.index();

    BlockVector<double> x(4), y(4);

    testMatrix(bcrsMatrix, x, y);

    // Test whether matrix resizing works
    int size = 3;
    bcrsMatrix.setSize(size,size,size);

    for (int i=0; i<size; i++)
      bcrsMatrix.setrowsize(i, 1);

    bcrsMatrix.endrowsizes();

    for (int i=0; i<size; i++)
      bcrsMatrix.addindex(i, i);

    bcrsMatrix.endindices();

    for (int i=0; i<size; i++)
      bcrsMatrix[i][i] = 1.0;

    x.resize(size);
    y.resize(size);
    testMatrix(bcrsMatrix, x, y);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class with FieldMatrix entries
  // ////////////////////////////////////////////////////////////

  BCRSMatrix<FieldMatrix<double,2,2> > bcrsMatrix(4,4, BCRSMatrix<FieldMatrix<double,2,2> >::random);

  bcrsMatrix.setrowsize(0,2);
  bcrsMatrix.setrowsize(1,3);
  bcrsMatrix.setrowsize(2,3);
  bcrsMatrix.setrowsize(3,2);

  bcrsMatrix.endrowsizes();

  bcrsMatrix.addindex(0, 0);
  bcrsMatrix.addindex(0, 1);

  bcrsMatrix.addindex(1, 0);
  bcrsMatrix.addindex(1, 1);
  bcrsMatrix.addindex(1, 2);

  bcrsMatrix.addindex(2, 1);
  bcrsMatrix.addindex(2, 2);
  bcrsMatrix.addindex(2, 3);

  bcrsMatrix.addindex(3, 2);
  bcrsMatrix.addindex(3, 3);

  bcrsMatrix.endindices();

  typedef BCRSMatrix<FieldMatrix<double,2,2> >::RowIterator RowIterator;
  typedef BCRSMatrix<FieldMatrix<double,2,2> >::ColIterator ColIterator;

  for(RowIterator row = bcrsMatrix.begin(); row != bcrsMatrix.end(); ++row)
    for(ColIterator col = row->begin(); col != row->end(); ++col)
      *col = 1.0 + (double) row.index() * (double) col.index();

  testSuperMatrix(bcrsMatrix);

  // Test whether matrix resizing works
  int size = 3;
  bcrsMatrix.setSize(size,size,size);

  for (int i=0; i<size; i++)
    bcrsMatrix.setrowsize(i, 1);

  bcrsMatrix.endrowsizes();

  for (int i=0; i<size; i++)
    bcrsMatrix.addindex(i, i);

  bcrsMatrix.endindices();

  for (int i=0; i<size; i++)
    bcrsMatrix[i][i] = 1.0;

  testSuperMatrix(bcrsMatrix);

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
    testSolve<BDMatrix<double>, BlockVector<double> >(bdMatrix3);

    // test whether resizing works
    bdMatrix2.setSize(5);
    bdMatrix2 = 4.0;
    x.resize(5);
    y.resize(5);
    testMatrix(bdMatrix2, x, y);

    // Test whether inversion works
    bdMatrix2.invert();
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
  testSolve<BDMatrix<FieldMatrix<double,2,2> >, BlockVector<FieldVector<double,2> > >(bdMatrix2);

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

    testSolve<BTDMatrix<double>, BlockVector<double> >(btdMatrixScalar);

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

  testSolve<BTDMatrix<FieldMatrix<double,1,1> >, BlockVector<FieldVector<double,1> > >(btdMatrixScalar);

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

  testSolve<BTDMatrix<FieldMatrix<double,2,2> >, BlockVector<FieldVector<double,2> > >(btdMatrix);

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

  FieldVector<double,1> dMatrixConstructFrom;
  dMatrixConstructFrom = 3.1459;

  DiagonalMatrix<double,4> dMatrix1;
  dMatrix1 = 3.1459;
  testMatrix(dMatrix1, fvX, fvY);

  DiagonalMatrix<double,4> dMatrix2(3.1459);
  testMatrix(dMatrix2, fvX, fvY);

  DiagonalMatrix<double,4> dMatrix3(dMatrixConstructFrom);
  testMatrix(dMatrix3, fvX, fvY);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the ScaledIdentityMatrix class
  // ////////////////////////////////////////////////////////////////////////

  ScaledIdentityMatrix<double,4> sIdMatrix;
  sIdMatrix = 3.1459;

  testMatrix(sIdMatrix, fvX, fvY);
}
