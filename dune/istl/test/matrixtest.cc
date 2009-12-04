// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/** \file
    \brief Unit tests for the different dynamic matrices provided by ISTL
 */
#include "config.h"
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/istl/scaledidmatrix.hh>
#include <dune/istl/diagonalmatrix.hh>


using namespace Dune;

template <class MatrixType>
void testSuperMatrix(MatrixType& matrix)
{
  // ////////////////////////////////////////////////////////
  //   Check the types which are exported by the matrix
  // ////////////////////////////////////////////////////////

  typedef typename MatrixType::field_type field_type;

  typedef typename MatrixType::block_type block_type;

  typedef typename MatrixType::size_type size_type;

  typedef typename MatrixType::allocator_type allocator_type;

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

  typedef typename MatrixType::block_type block_type;

  typedef typename MatrixType::row_type row_type;

  typedef typename MatrixType::size_type size_type;

  typedef typename MatrixType::RowIterator RowIterator;

  typedef typename MatrixType::ConstRowIterator ConstRowIterator;

  typedef typename MatrixType::ColIterator ColIterator;

  typedef typename MatrixType::ConstColIterator ConstColIterator;

  assert(MatrixType::blocklevel >= 0);

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

  rowIt    = matrix.rbegin();
  rowEndIt = matrix.rend();

  numRows    = 0;
  numEntries = 0;

  for (; rowIt!=rowEndIt; --rowIt) {

    typename MatrixType::ColIterator colIt    = rowIt->rbegin();
    typename MatrixType::ColIterator colEndIt = rowIt->rend();

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

  constRowIt    = matrix.rbegin();
  constRowEndIt = matrix.rend();

  numRows    = 0;
  numEntries = 0;

  for (; constRowIt!=constRowEndIt; --constRowIt) {

    typename MatrixType::ConstColIterator constColIt    = constRowIt->rbegin();
    typename MatrixType::ConstColIterator constColEndIt = constRowIt->rend();

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
  MatrixType thirdMatrix(matrix);

  // ///////////////////////////////////////////////////////
  //   Test component-wise operations
  // ///////////////////////////////////////////////////////

  matrix *= M_PI;
  matrix /= M_PI;

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

  matrix.usmv(M_PI,x,y);

  matrix.usmtv(M_PI,x,y);

  matrix.usmhv(M_PI,x,y);

  // //////////////////////////////////////////////////////////////
  //   Test the matrix norms
  // //////////////////////////////////////////////////////////////

  double frobenius_norm = matrix.frobenius_norm();

  frobenius_norm += matrix.frobenius_norm2();

  frobenius_norm += matrix.infinity_norm();

  frobenius_norm += matrix.infinity_norm_real();

}




int main()
{
  // ////////////////////////////////////////////////////////////
  //   Test the Matrix class -- a dense dynamic matrix
  // ////////////////////////////////////////////////////////////

  Matrix<FieldMatrix<double,3,3> > matrix(10,10);
  for (int i=0; i<10; i++)
    for (int j=0; j<10; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          matrix[i][j][k][l] = (i+j)/((double)(k*l));            // just anything

  testSuperMatrix(matrix);

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class -- a sparse dynamic matrix
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

  // ////////////////////////////////////////////////////////////////////////
  //   Test the BDMatrix class -- a dynamic block-diagonal matrix
  // ////////////////////////////////////////////////////////////////////////

  BDMatrix<FieldMatrix<double,4,4> > bdMatrix(2);
  bdMatrix = 4.0;

  testSuperMatrix(bdMatrix);

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
  //   Test the ScaledIdentityMatrix class
  // ////////////////////////////////////////////////////////////////////////

  DiagonalMatrix<double,4> dMatrix;
  dMatrix = 3.1459;

  testMatrix(dMatrix, fvX, fvY);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the ScaledIdentityMatrix class
  // ////////////////////////////////////////////////////////////////////////

  ScaledIdentityMatrix<double,4> sIdMatrix;
  sIdMatrix = 3.1459;

  testMatrix(sIdMatrix, fvX, fvY);
}
