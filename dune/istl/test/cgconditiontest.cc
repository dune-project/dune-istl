// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>

typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > MAT;
typedef Dune::BlockVector<Dune::FieldVector<double,1> > VEC;

void condition_test (MAT& T) {

  int N = T.N();

  // CG run (noop preconditioner) with condition estimate

  auto prec = std::make_shared<Dune::Richardson<VEC, VEC>> (1.0);

  auto op = std::make_shared<Dune::MatrixAdapter<MAT, VEC, VEC>>(T);

  VEC d (N,N);
  VEC v (N,N);

  v = 1.0;
  d = 1.0;

  int verbosity = 2;

  Dune::InverseOperatorResult result;
  auto solver = std::make_shared<Dune::CGSolver<VEC> >(*op,*prec,1E-6,1000,verbosity,true);
  solver->apply(v,d,result);



#if HAVE_ARPACKPP
  // Actual condition number using ARPACK
  Dune::ArPackPlusPlus_Algorithms<MAT, VEC> arpack(T);

  double eps = 0.0;
  VEC eigv;
  double min_eigv, max_eigv;
  arpack.computeSymMinMagnitude (eps, eigv, min_eigv);
  arpack.computeSymMaxMagnitude (eps, eigv, max_eigv);

  std::cout << "Actual condition number from ARPACK: " << max_eigv / min_eigv << std::endl;
#endif
}



int main(int argc, char **argv)
{

  // Simple stencil
  {
    int N = 142;

    MAT T(N, N, MAT::row_wise);

    for (auto row = T.createbegin(); row != T.createend(); ++row) {
      if (row.index() > 0)
        row.insert(row.index()-1);
      row.insert(row.index());
      if (row.index() < T.N() - 1)
        row.insert(row.index()+1);
    }
    for (int row = 0; row < N; ++row) {
      T[row][row] = 2.0;
      if (row > 0)
        T[row][row-1] = -1.0;
      if (row < N-1)
        T[row][row+1] = -1.0;
    }

    condition_test (T);
  }


  // Ill-conditioned stencil
  {
    int N = 142;

    MAT T(N, N, MAT::row_wise);

    for (auto row = T.createbegin(); row != T.createend(); ++row) {
      if (row.index() > 0)
        row.insert(row.index()-1);
      row.insert(row.index());
      if (row.index() < T.N() - 1)
        row.insert(row.index()+1);
    }
    for (int row = 0; row < N; ++row) {
      double factor = 0.8 + 0.2 * ((double)row / N);
      T[row][row] = 2.0 * factor;
      if (row > 0)
        T[row][row-1] = -1.0 * factor;
      if (row < N-1)
        T[row][row+1] = -1.0 * factor;
    }

    condition_test (T);
  }


  // Very ill-conditioned stencil
  {
    int N = 40;

    MAT T(N, N, MAT::row_wise);

    for (auto row = T.createbegin(); row != T.createend(); ++row) {
      for (int i = -10; i <= 10; i++) {
        if (row.index()+i >= 0 && row.index()+i < T.N())
          row.insert(row.index()+i);
      }
    }
    for (int row = 0; row < N; ++row) {
      double factor = 0.8 + 0.2 * ((double)row / N);
      for (int i = -10; i <= 10; i++) {
        if (row+i >= 0 && row+i < int(T.N()))
          T[row][row+i] = -1.0 * factor;
      }
      T[row][row] = 20.0 * factor;
    }

    condition_test (T);
  }

}
