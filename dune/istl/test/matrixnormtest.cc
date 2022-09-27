// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <complex>
#include <memory>

#include <dune/common/fmatrix.hh>
#include <dune/common/classname.hh>
#include <dune/istl/matrix.hh>

template <class V>
void checkNormNANVector(V const &v, int line) {
  if (!std::isnan(v.infinity_norm())) {
    std::cerr << "error: norm not NaN: infinity_norm() on line " << line
              << " (type: " << Dune::className(v[0]) << ")" << std::endl;
    std::exit(-1);
  }
}

template <class M>
void checkNormNANMatrix(M const &v, int line) {
  if (!std::isnan(v.frobenius_norm())) {
    std::cerr << "error: norm not NaN: frobenius_norm() on line " << line
              << " (type: " << Dune::className(v[0][0]) << ")" << std::endl;
    std::exit(-1);
  }
  if (!std::isnan(v.infinity_norm())) {
    std::cerr << "error: norm not NaN: infinity_norm() on line " << line
              << " (type: " << Dune::className(v[0][0]) << ")" << std::endl;
    std::exit(-1);
  }
}

// Make sure that matrices with NaN entries have norm NaN.
// See also bug flyspray/FS#1147
template <typename T>
void test_nan(T const &mynan) {
  using M = Dune::Matrix<Dune::FieldMatrix<T, 2, 2>>;
  T n(0);
  {
    M m(2, 2);
    m[0][0] = {{n, n}, {n, mynan}};
    m[0][1] = n;
    m[1][0] = n;
    m[1][1] = n;
    checkNormNANVector(m[0], __LINE__);
    checkNormNANMatrix(m, __LINE__);
  }
  {
    M m(2, 2);
    m[0][0] = n;
    m[0][1] = {{n, n}, {n, mynan}};
    m[1][0] = n;
    m[1][1] = n;
    checkNormNANVector(m[0], __LINE__);
    checkNormNANMatrix(m, __LINE__);
  }
  {
    M m(2, 2);
    m[0][0] = n;
    m[0][1] = n;
    m[1][0] = {{n, n}, {n, mynan}};
    m[1][1] = n;
    checkNormNANVector(m[1], __LINE__);
    checkNormNANMatrix(m, __LINE__);
  }
}

int main() {
  {
    double nan = std::nan("");
    test_nan(nan);
  }
  {
    std::complex<double> nan(std::nan(""), 17);
    test_nan(nan);
  }
}
