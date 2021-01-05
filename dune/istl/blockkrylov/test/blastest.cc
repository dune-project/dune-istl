// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <config.h>

#include <array>

#include "dune/istl/blockkrylov/blas.hh"

int main(){
  std::array<std::complex<double>, 16> a, b, c; // 4x4 matrices

  std::complex<double> alpha, beta;
  int n=4;

  Dune::BLAS::gemm("T", "T", &n,&n,&n,&alpha,
                   a.data(),&n,
                   b.data(), &n, &beta,
                   c.data(), &n);

  return 0;
}
