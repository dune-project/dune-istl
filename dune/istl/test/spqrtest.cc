// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>
#include <complex>
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/spqr.hh>

#include "laplacian.hh"

template <class FIELD_TYPE, int BS>
void run(std::size_t N)
{
#if HAVE_SUITESPARSE_SPQR
  std::cout << "testing for N=" << N << " BS=" << BS << std::endl;

  typedef Dune::FieldMatrix<FIELD_TYPE,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<FIELD_TYPE,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  Vector b(N*N), x(N*N);

  setupLaplacian(mat,N);
  b=1;
  x=0;

  Dune::Timer watch;

  watch.reset();

  Dune::SPQR<BCRSMat> solver(mat,1);

  Dune::InverseOperatorResult res;

  Dune::SPQR<BCRSMat> solver1;

  std::set<std::size_t> mrs;
  for(std::size_t s=0; s < N/2; ++s)
    mrs.insert(s);

  solver1.setSubMatrix(mat,mrs);
  solver1.setVerbosity(true);

  solver.apply(x,b,res);
  solver.free();

  Vector residuum(N*N);
  residuum=0;
  fop.apply(x,residuum);
  residuum-=b;
  std::cout<<"Residuum : "<<residuum.two_norm()<<std::endl;

  solver1.apply(x,b,res);
#endif
}

int main(int argc, char** argv)
{
#if HAVE_SUITESPARSE_SPQR
  try
  {
    std::size_t N=100;
    if (argc > 1)
      N = atoi(argv[1]);

    run<double,1>(N);
    run<double,2>(N);
    // run<std::complex<double>,1>(N); // does not work
    // run<std::complex<double>,2>(N);

    return 0;
  }
  catch(Dune::Exception &e)
  {
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...)
  {
    std::cerr << "Unknown exception" << std::endl;
  }
#else // HAVE_SUITESPARSE_SPQR
  std::cerr << "You need SuiteSparse's SPQR to run this test." << std::endl;
  return 77;
#endif // HAVE_SUITESPARSE_SPQR
}
