// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <complex>
#include <memory>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

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

template <typename T>
std::shared_ptr<Dune::BCRSMatrix<Dune::FieldMatrix<T, 2, 2>>> genPattern() {
  using LocalMatrix = Dune::FieldMatrix<T, 2, 2>;
  using GlobalMatrix = Dune::BCRSMatrix<LocalMatrix>;

  // Build a 3x3 matrix with sparsity pattern
  //
  // +-+
  // ---
  // +-+
  auto m = std::make_shared<GlobalMatrix>(3, 3, GlobalMatrix::random);

  m->setrowsize(0, 2);
  m->setrowsize(1, 0);
  m->setrowsize(2, 2);
  m->endrowsizes();

  m->addindex(0, 0);
  m->addindex(0, 2);
  m->addindex(2, 0);
  m->addindex(2, 2);
  m->endindices();

  return m;
}

// Make sure that matrices with NaN entries have norm NaN.
// See also bug flyspray/FS#1147
template <typename T>
void test_nan(T const &mynan) {
    T n(0);
    {
      auto m = genPattern<T>();
      (*m)[0][0] = {{n, n}, {n, mynan}};
      (*m)[0][2] = n;
      (*m)[2][0] = n;
      (*m)[2][2] = n;
      checkNormNANVector((*m)[0], __LINE__);
      checkNormNANMatrix(*m, __LINE__);
    }
    {
      auto m = genPattern<T>();
      (*m)[0][0] = n;
      (*m)[0][2] = {{n, n}, {n, mynan}};;
      (*m)[2][0] = n;
      (*m)[2][2] = n;
      checkNormNANVector((*m)[0], __LINE__);
      checkNormNANMatrix(*m, __LINE__);
    }
    {
      auto m = genPattern<T>();
      (*m)[0][0] = n;
      (*m)[0][2] = n;
      (*m)[2][0] = {{n, n}, {n, mynan}};;
      (*m)[2][2] = n;
      checkNormNANVector((*m)[2], __LINE__);
      checkNormNANMatrix(*m, __LINE__);
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
