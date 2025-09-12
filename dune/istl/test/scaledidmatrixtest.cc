// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// include this first to see whether it includes all necessary headers itself
#include <dune/istl/scaledidmatrix.hh>

#include <iostream>
#include <algorithm>

#include <dune/common/diagonalmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>

using namespace Dune;


template<class K, int n>
TestSuite test_matrix()
{
  TestSuite test("test_matrix");

  ScaledIdentityMatrix<K,n> A(1);

  // Test for proper matrix sizes, both statically and dynamically
  static_assert(A.rows==n, "Incorrect number of rows");
  static_assert(A.cols==n, "Incorrect number of columns");

  test.check(A.rows==n) << "Incorrect number of rows";
  test.check(A.cols==n) << "Incorrect number of columns";

  test.check(A.rows()==n) << "Incorrect number of rows";
  test.check(A.cols()==n) << "Incorrect number of columns";

  FieldVector<K,n> f;
  FieldVector<K,n> v;

  // assign matrix
  A=2;

  // assign vector
  f = 1;
  v = 2;

  // matrix vector product
  A.umv(v,f);

  // matrix times scalar
  [[maybe_unused]] auto r1 = A * 23.;
  [[maybe_unused]] auto r2 = 42. * A;

  // test norms
  A.frobenius_norm();
  A.frobenius_norm2();
  A.infinity_norm();
  A.infinity_norm_real();

  // print matrix
  std::cout << A << std::endl;
  // print vector
  std::cout << f << std::endl;

  // Construction of FieldMatrix from ScaledIdentityMatrix
  FieldMatrix<K,n,n> AFM = FieldMatrix<K,n,n>(A);

  // Test whether we can add a ScaledIdentityMatrix to a FieldMatrix,
  // and vice versa
  const auto sum = A + AFM + A;

  // Is the sum correctly computed?
  for (std::size_t i=0; i<sum.N(); ++i)
    for (std::size_t j=0; j<sum.M(); ++j)
      if (i==j)
        test.check(sum[i][j] == A[i][j] + AFM[i][j] + A[i][j]) << "sum diagonal";
      else // Off-diagonal entries of a ScaledIdentityMatrix cannot be accessed.
        test.check(sum[i][j] == AFM[i][j]) << "sum off-diagonal";

  // Construct a number type different from K
  using OtherScalar = std::conditional_t<std::is_same_v<K,float>, double, float>;

  // Construction of FieldMatrix from ScaledIdentityMatrix
  auto AFMOtherScalar = FieldMatrix<OtherScalar,n,n>(A);

  // Test whether we can add a ScaledIdentityMatrix to a FieldMatrix
  // with a different number type, and vice versa
  const auto sum2 = A + AFMOtherScalar + A;
  AFMOtherScalar += A;

  // Is the sum correctly computed?
  for (std::size_t i=0; i<sum2.N(); ++i)
  {
    for (std::size_t j=0; j<sum2.M(); ++j)
      if (i==j)
        test.check(sum2[i][j] == A[i][j] + AFMOtherScalar[i][j]) << "sum2 diagonal";
      else // Off-diagonal entries of a ScaledIdentityMatrix cannot be accessed.
        test.check(sum2[i][j] == AFMOtherScalar[i][j]) << "sum2 off-diagonal";
  }

  // Construction of DiagonalMatrix from ScaledIdentityMatrix
  auto ADMOtherScalar = DiagonalMatrix<OtherScalar,n>(42.0);

  // Test whether we can add a ScaledIdentityMatrix to a DiagonalMatrix
  // with a different number type, and vice versa
  const auto sum3 = A + ADMOtherScalar + A;
  ADMOtherScalar += A;

  // Is the sum correctly computed?
  for (std::size_t i=0; i<sum3.N(); ++i)
    test.check(sum3[i][i] == A[i][i] + ADMOtherScalar[i][i]) << "sum3 diagonal";

  return test;
}

int main()
{
  TestSuite test("ScaledIdentityMatrix");

  test.subTest(test_matrix<float, 1>());
  test.subTest(test_matrix<double, 1>());
  //test.subTest(test_matrix<int, 10>()); Does not compile with icc because there is no std::sqrt(int)  std::fabs(int)
  test.subTest(test_matrix<double, 5>());

  return test.exit();
}
