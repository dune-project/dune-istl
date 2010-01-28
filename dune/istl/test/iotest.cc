// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include "laplacian.hh"

int main(int argc, char** argv)
{
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > Matrix;
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<std::complex<double>,1,1> > ComplexMatrix;

  Matrix A;
  ComplexMatrix C;

  setupLaplacian(A, 3);
  setupLaplacian(C, 3);

  C[0][0]=std::complex<double>(0,-1);

  writeMatrixToMatlabHelper(A, 0, 0, std::cout);
  writeMatrixToMatlabHelper(C, 0, 0, std::cout);

}
