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


using namespace Dune;

template <class MatrixType>
void testMatrix(MatrixType& matrix)
{
  // ////////////////////////////////////////////////////////
  //   Check the types which are exported by the matrix
  // ////////////////////////////////////////////////////////

  typedef typename MatrixType::field_type field_type;

  typedef typename MatrixType::block_type block_type;

  typedef typename MatrixType::allocator_type allocator_type;

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

  int numRows = 0, numEntries = 0;

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

  size_type n = matrix.N();
  size_type m = matrix.M();

  size_type rowdim = matrix.rowdim();
  size_type coldim = matrix.coldim();

  // the last two, but for specific rows/columns
  rowdim = matrix.rowdim(0);
  coldim = matrix.coldim(0);

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

  BlockVector<FieldVector<field_type, block_type::cols> > x(matrix.M());
  BlockVector<FieldVector<field_type, block_type::rows> > y(matrix.N());

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

  double frobenius_norm2 = matrix.frobenius_norm2();

  double infinity_norm   = matrix.infinity_norm();

  double infinity_norm_real = matrix.infinity_norm_real();

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

  testMatrix(matrix);

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

  testMatrix(bcrsMatrix);

  // ////////////////////////////////////////////////////////////////////////
  //   Test the BDMatrix class -- a dynamic block-diagonal matrix
  // ////////////////////////////////////////////////////////////////////////

  BDMatrix<FieldMatrix<double,4,4> > bdMatrix(2);
  bdMatrix = 4.0;

  testMatrix(bdMatrix);
}
