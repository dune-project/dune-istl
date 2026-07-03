// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

/** \file \brief Tests for directional Gauss-Seidel and related block solves.
 */

#include <array>
#include <cmath>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/gsetc.hh>

namespace {

using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

constexpr double eps = 1e-12;
constexpr double weight = 0.5;

void
checkClose(Dune::TestSuite& t, const Vector& actual, const Vector& expected, const char* message)
{
  Vector diff(actual);
  diff -= expected;
  t.check(diff.two_norm() <= eps)
    << message << ", difference norm = " << diff.two_norm();
}

Dune::TestSuite
gaussSeidelDense3x3DirectionalSweeps()
{
  Dune::TestSuite t;

  Matrix A(3, 3, 9, Matrix::row_wise);
  for (auto row = A.createbegin(); row != A.createend(); ++row) {
    row.insert(0);
    row.insert(1);
    row.insert(2);
  }

  A[0][0][0][0] = 4.0; A[0][1][0][0] =  1.0; A[0][2][0][0] =  2.0;
  A[1][0][0][0] = 1.0; A[1][1][0][0] =  3.0; A[1][2][0][0] = -1.0;
  A[2][0][0][0] = 2.0; A[2][1][0][0] = -1.0; A[2][2][0][0] =  5.0;

  Vector d(3), xInit(3);
  d[0]     = 7.0; d[1]     = -4.0; d[2]     = 6.0;
  xInit[0] = 1.0; xInit[1] = -2.0; xInit[2] = 0.5;

  // Hand-computed forward GS sweep:
  Vector xForwardUpdate(3);
  xForwardUpdate[0][0] = (d[0][0] - A[0][1][0][0]*xInit[1][0] - A[0][2][0][0]*xInit[2][0])/A[0][0][0][0] - xInit[0][0];
  xForwardUpdate[1][0] = (d[1][0] - A[1][0][0][0]*(xInit[0][0] + weight * xForwardUpdate[0][0]) - A[1][2][0][0]*xInit[2][0])/A[1][1][0][0] - xInit[1][0];
  xForwardUpdate[2][0] = (d[2][0] - A[2][0][0][0]*(xInit[0][0] + weight * xForwardUpdate[0][0]) - A[2][1][0][0]*(xInit[1][0] + weight * xForwardUpdate[1][0]))/A[2][2][0][0] - xInit[2][0];
  Vector xForwardCheck(3);
  xForwardCheck[0][0] = xInit[0][0] + weight * xForwardUpdate[0][0];
  xForwardCheck[1][0] = xInit[1][0] + weight * xForwardUpdate[1][0];
  xForwardCheck[2][0] = xInit[2][0] + weight * xForwardUpdate[2][0];

  Vector xForward(xInit);
  Dune::bsorf(A, xForward, d, weight);

  checkClose(t, xForward, xForwardCheck,
             "Error in bsorf, does not match 3x3 hand-computed forward sweep");

  // Hand-computed backward GS sweep:
  Vector xBackwardUpdate(3);
  xBackwardUpdate[2][0] = (d[2][0] - A[2][0][0][0]*xInit[0][0] - A[2][1][0][0]*xInit[1][0])/A[2][2][0][0] - xInit[2][0];
  xBackwardUpdate[1][0] = (d[1][0] - A[1][0][0][0]*xInit[0][0] - A[1][2][0][0]*(xInit[2][0] + weight * xBackwardUpdate[2][0]))/A[1][1][0][0] - xInit[1][0];
  xBackwardUpdate[0][0] = (d[0][0] - A[0][1][0][0]*(xInit[1][0] + weight * xBackwardUpdate[1][0]) - A[0][2][0][0]*(xInit[2][0] + weight * xBackwardUpdate[2][0]))/A[0][0][0][0] - xInit[0][0];
  Vector xBackwardCheck(3);
  xBackwardCheck[2][0] = xInit[2][0] + weight * xBackwardUpdate[2][0];
  xBackwardCheck[1][0] = xInit[1][0] + weight * xBackwardUpdate[1][0];
  xBackwardCheck[0][0] = xInit[0][0] + weight * xBackwardUpdate[0][0];

  Vector xBackward(xInit);
  Dune::bsorb(A, xBackward, d, weight);

  checkClose(t, xBackward, xBackwardCheck,
             "Error in bsorb, does not match 3x3 hand-computed backward sweep");

  return t;
}

Dune::TestSuite
gaussSeidelSparse5x5DirectionalSweeps()
{
  Dune::TestSuite t;

  Matrix A(5, 5, 13, Matrix::row_wise);
  for (auto row = A.createbegin(); row != A.createend(); ++row) {
    if (row.index() == 0) {
      row.insert(0);
      row.insert(1);
    } else if (row.index() == 1) {
      row.insert(0);
      row.insert(1);
      row.insert(2);
    } else if (row.index() == 2) {
      row.insert(1);
      row.insert(2);
      row.insert(3);
    } else if (row.index() == 3) {
      row.insert(2);
      row.insert(3);
      row.insert(4);
    } else {
      row.insert(3);
      row.insert(4);
    }
  }

  A[0][0] = 4.0; A[0][1] = -1.0;
  A[1][0] = 2.0; A[1][1] =  5.0; A[1][2] = 1.0;
                 A[2][1] =  3.0; A[2][2] = 6.0; A[2][3] = -2.0;
                                 A[3][2] = 1.0; A[3][3] =  7.0; A[3][4] = 2.0;
                                                A[4][3] = -1.0; A[4][4] = 8.0;



  Vector d(5), xInit(5);
  d[0]     = 5.0;  d[1]     = -3.0; d[2]     =  4.0; d[3]     = 10.0; d[4]     = -2.0;
  xInit[0] = 1.0;  xInit[1] = 0.0;  xInit[2] = -1.0; xInit[3] = 2.0;  xInit[4] =  0.5;

  // Hand-computed forward GS sweep (entry-by-entry).
  Vector xForwardUpdate(5);
  xForwardUpdate[0][0] = (d[0][0] - A[0][1][0][0]*xInit[1][0])/A[0][0][0][0] - xInit[0][0];
  xForwardUpdate[1][0] = (d[1][0] - A[1][0][0][0]*(xInit[0][0] + weight * xForwardUpdate[0][0]) - A[1][2][0][0]*xInit[2][0])/A[1][1][0][0] - xInit[1][0];
  xForwardUpdate[2][0] = (d[2][0] - A[2][1][0][0]*(xInit[1][0] + weight * xForwardUpdate[1][0]) - A[2][3][0][0]*xInit[3][0])/A[2][2][0][0] - xInit[2][0];
  xForwardUpdate[3][0] = (d[3][0] - A[3][2][0][0]*(xInit[2][0] + weight * xForwardUpdate[2][0]) - A[3][4][0][0]*xInit[4][0])/A[3][3][0][0] - xInit[3][0];
  xForwardUpdate[4][0] = (d[4][0] - A[4][3][0][0]*(xInit[3][0] + weight * xForwardUpdate[3][0]))/A[4][4][0][0] - xInit[4][0];
  Vector xForwardCheck(xInit);
  xForwardCheck[0][0] = xInit[0][0] + weight * xForwardUpdate[0][0];
  xForwardCheck[1][0] = xInit[1][0] + weight * xForwardUpdate[1][0];
  xForwardCheck[2][0] = xInit[2][0] + weight * xForwardUpdate[2][0];
  xForwardCheck[3][0] = xInit[3][0] + weight * xForwardUpdate[3][0];
  xForwardCheck[4][0] = xInit[4][0] + weight * xForwardUpdate[4][0];

  Vector xForward(xInit);
  Dune::bsorf(A, xForward, d, weight);

  checkClose(t, xForward, xForwardCheck,
             "Error in bsorf, does not match 5x5 hand-computed forward sweep");

  // Hand-computed backward GS sweep (entry-by-entry).
  Vector xBackwardUpdate(5);
  xBackwardUpdate[4][0] = (d[4][0] - A[4][3][0][0]*xInit[3][0])/A[4][4][0][0] - xInit[4][0];
  xBackwardUpdate[3][0] = (d[3][0] - A[3][2][0][0]*xInit[2][0] - A[3][4][0][0]*(xInit[4][0] + weight * xBackwardUpdate[4][0]))/A[3][3][0][0] - xInit[3][0];
  xBackwardUpdate[2][0] = (d[2][0] - A[2][1][0][0]*xInit[1][0] - A[2][3][0][0]*(xInit[3][0] + weight * xBackwardUpdate[3][0]))/A[2][2][0][0] - xInit[2][0];
  xBackwardUpdate[1][0] = (d[1][0] - A[1][0][0][0]*xInit[0][0] - A[1][2][0][0]*(xInit[2][0] + weight * xBackwardUpdate[2][0]))/A[1][1][0][0] - xInit[1][0];
  xBackwardUpdate[0][0] = (d[0][0] - A[0][1][0][0]*(xInit[1][0] + weight * xBackwardUpdate[1][0]))/A[0][0][0][0] - xInit[0][0];
  Vector xBackwardCheck(xInit);
  xBackwardCheck[4][0] = xInit[4][0] + weight * xBackwardUpdate[4][0];
  xBackwardCheck[3][0] = xInit[3][0] + weight * xBackwardUpdate[3][0];
  xBackwardCheck[2][0] = xInit[2][0] + weight * xBackwardUpdate[2][0];
  xBackwardCheck[1][0] = xInit[1][0] + weight * xBackwardUpdate[1][0];
  xBackwardCheck[0][0] = xInit[0][0] + weight * xBackwardUpdate[0][0];

  Vector xBackward(xInit);
  Dune::bsorb(A, xBackward, d, weight);
  checkClose(t, xBackward, xBackwardCheck,
             "Error in bsorb, does not match 5x5 hand-computed backward sweep");

  return t;
}

Dune::TestSuite
blockDiagonalDense3x3Solve()
{
  Dune::TestSuite t;

  Matrix A(3, 3, 9, Matrix::row_wise);
  for (auto row = A.createbegin(); row != A.createend(); ++row) {
    row.insert(0);
    row.insert(1);
    row.insert(2);
  }

  A[0][0] = 4.0; A[0][1] =  1.0; A[0][2] =  2.0;
  A[1][0] = 1.0; A[1][1] =  3.0; A[1][2] = -1.0;
  A[2][0] = 2.0; A[2][1] = -1.0; A[2][2] =  5.0;

  Vector d(3);
  d[0] = 7.0; d[1] = -4.0; d[2] = 6.0;

  Vector diagonalCheck(3);
  diagonalCheck[0][0] = weight * d[0][0] / A[0][0][0][0];
  diagonalCheck[1][0] = weight * d[1][0] / A[1][1][0][0];
  diagonalCheck[2][0] = weight * d[2][0] / A[2][2][0][0];

  Vector diagonal(3);
  Dune::bdsolve(A, diagonal, d, weight);
  checkClose(t, diagonal, diagonalCheck,
             "Error in bdsolve, does not match 3x3 hand-computed diagonal solve");

  return t;
}

Dune::TestSuite
blockTriangularAndDiagonalDense3x3Solves()
{
  Dune::TestSuite t;

  Matrix A(3, 3, 9, Matrix::row_wise);
  for (auto row = A.createbegin(); row != A.createend(); ++row) {
    row.insert(0);
    row.insert(1);
    row.insert(2);
  }

  A[0][0] = 4.0; A[0][1] =  1.0; A[0][2] =  2.0;
  A[1][0] = 1.0; A[1][1] =  3.0; A[1][2] = -1.0;
  A[2][0] = 2.0; A[2][1] = -1.0; A[2][2] =  5.0;

  Vector d(3);
  d[0] = 7.0; d[1] = -4.0; d[2] = 6.0;

  // Hand-computed solve for (L + D) v = d.
  Vector lowerCheck(3);
  lowerCheck[0][0] = weight * d[0][0] / A[0][0][0][0];
  lowerCheck[1][0] = weight * (d[1][0] - A[1][0][0][0] * lowerCheck[0][0]) / A[1][1][0][0];
  lowerCheck[2][0] = weight * (d[2][0] - A[2][0][0][0] * lowerCheck[0][0] - A[2][1][0][0] * lowerCheck[1][0]) / A[2][2][0][0];

  Vector lower(3);
  Dune::bltsolve(A, lower, d, weight);
  checkClose(t, lower, lowerCheck,
             "Error in bltsolve, does not match 3x3 hand-computed lower triangular solve");

  // Hand-computed solve for (D + U) v = d.
  Vector upperCheck(3);
  upperCheck[2][0] = weight * d[2][0] / A[2][2][0][0];
  upperCheck[1][0] = weight * (d[1][0] - A[1][2][0][0] * upperCheck[2][0]) / A[1][1][0][0];
  upperCheck[0][0] = weight * (d[0][0] - A[0][1][0][0] * upperCheck[1][0] - A[0][2][0][0] * upperCheck[2][0]) / A[0][0][0][0];

  Vector upper(3);
  Dune::butsolve(A, upper, d, weight);
  checkClose(t, upper, upperCheck,
             "Error in butsolve, does not match 3x3 hand-computed upper triangular solve");

  return t;
}

Dune::TestSuite
blockTriangularAndDiagonalSparse5x5Solves()
{
  Dune::TestSuite t;

  Matrix A(5, 5, 13, Matrix::row_wise);
  for (auto row = A.createbegin(); row != A.createend(); ++row) {
    if (row.index() == 0) {
      row.insert(0);
      row.insert(1);
    } else if (row.index() == 1) {
      row.insert(0);
      row.insert(1);
      row.insert(2);
    } else if (row.index() == 2) {
      row.insert(1);
      row.insert(2);
      row.insert(3);
    } else if (row.index() == 3) {
      row.insert(2);
      row.insert(3);
      row.insert(4);
    } else {
      row.insert(3);
      row.insert(4);
    }
  }

  A[0][0][0][0] = 4.0; A[0][1][0][0] = -1.0;
  A[1][0][0][0] = 2.0; A[1][1][0][0] =  5.0; A[1][2][0][0] = 1.0;
                       A[2][1][0][0] =  3.0; A[2][2][0][0] = 6.0; A[2][3][0][0] = -2.0;
                                             A[3][2][0][0] = 1.0; A[3][3][0][0] =  7.0; A[3][4][0][0] = 2.0;
                                                                  A[4][3][0][0] = -1.0; A[4][4][0][0] = 8.0;

  Vector d(5);
  d[0][0] = 5.0; d[1][0] = -3.0; d[2][0] = 4.0; d[3][0] = 10.0; d[4][0] = -2.0;

  // Hand-computed solve for (L + D) v = d.
  Vector lowerCheck(5);
  lowerCheck[0][0] = weight * d[0][0] / A[0][0][0][0];
  lowerCheck[1][0] = weight * (d[1][0] - A[1][0][0][0] * lowerCheck[0][0]) / A[1][1][0][0];
  lowerCheck[2][0] = weight * (d[2][0] - A[2][1][0][0] * lowerCheck[1][0]) / A[2][2][0][0];
  lowerCheck[3][0] = weight * (d[3][0] - A[3][2][0][0] * lowerCheck[2][0]) / A[3][3][0][0];
  lowerCheck[4][0] = weight * (d[4][0] - A[4][3][0][0] * lowerCheck[3][0]) / A[4][4][0][0];

  Vector lower(5);
  Dune::bltsolve(A, lower, d, weight);
  checkClose(t, lower, lowerCheck,
             "Error in bltsolve, does not match 5x5 hand-computed lower triangular solve");

  // Hand-computed solve for (D + U) v = d.
  Vector upperCheck(5);
  upperCheck[4][0] = weight * d[4][0] / A[4][4][0][0];
  upperCheck[3][0] = weight * (d[3][0] - A[3][4][0][0] * upperCheck[4][0]) / A[3][3][0][0];
  upperCheck[2][0] = weight * (d[2][0] - A[2][3][0][0] * upperCheck[3][0]) / A[2][2][0][0];
  upperCheck[1][0] = weight * (d[1][0] - A[1][2][0][0] * upperCheck[2][0]) / A[1][1][0][0];
  upperCheck[0][0] = weight * (d[0][0] - A[0][1][0][0] * upperCheck[1][0]) / A[0][0][0][0];

  Vector upper(5);
  Dune::butsolve(A, upper, d, weight);
  checkClose(t, upper, upperCheck,
             "Error in butsolve, does not match 5x5 hand-computed upper triangular solve");

  return t;
}

} // namespace

int
main()
{
  Dune::TestSuite t;
  t.subTest(gaussSeidelDense3x3DirectionalSweeps());
  t.subTest(gaussSeidelSparse5x5DirectionalSweeps());
  t.subTest(blockDiagonalDense3x3Solve());
  t.subTest(blockTriangularAndDiagonalDense3x3Solves());
  t.subTest(blockTriangularAndDiagonalSparse5x5Solves());
  return t.exit();
}
