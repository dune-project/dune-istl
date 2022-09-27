// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/fmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixmatrix.hh>

int main(int argc, char** argv)
{
  typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1> > MatrixType;
  MatrixType m1(2,2,MatrixType::random) ,
  m2(2,2,MatrixType::random) ,
  res(2,2,MatrixType::random);

  // initialize first matrix [1,0;0,1]
  m1.setrowsize(0,1);
  m1.setrowsize(1,1);
  m1.endrowsizes();
  m1.addindex(0,0);
  m1.addindex(1,1);
  m1.endindices();
  m1[0][0] = 1;
  m1[1][1] = 1;
  // initialize second matrix [0,1;1,0]
  m2.setrowsize(0,1);
  m2.setrowsize(1,1);
  m2.endrowsizes();
  m2.addindex(0,1);
  m2.addindex(1,0);
  m2.endindices();
  m2[0][1] = 1;
  m2[1][0] = 1;
  Dune::printmatrix(std::cout, m1, "m1", "");
  Dune::printmatrix(std::cout, m2, "m2", "");
  Dune::matMultTransposeMat(res, m1, m2);
  Dune::printmatrix(std::cout, res, "res", "");
  return 0;
}
