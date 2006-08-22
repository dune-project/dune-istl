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
  //   Count number of rows, columns, and nonzero entries
  // ////////////////////////////////////////////////////////

  typename MatrixType::RowIterator rowIt    = matrix.begin();
  typename MatrixType::RowIterator rowEndIt = matrix.end();

  int numRows = 0, numEntries = 0;

  for (; rowIt!=rowEndIt; ++rowIt) {

    typename MatrixType::ColIterator colIt    = rowIt->begin();
    typename MatrixType::ColIterator colEndIt = rowIt->end();

    for (; colIt!=colEndIt; ++colIt)
      numEntries++;

    numRows++;

  }

  assert (numRows == matrix.N());

  // test assignment from scalar
  matrix = 0;

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

}
