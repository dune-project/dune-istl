// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/test/iteratortest.hh>
#include <iostream>

class RowFunc {
public:
  void operator()(const Dune::FieldMatrix<double,1,1>& t){
    std::cout << t <<" ";
  }
};

class MatrixFunc
{
public:
  void operator()(const Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >::row_type& row)
  {
    std::cout << *(row.begin())<<" ";
  }
};

int main()
{
  using namespace Dune;

  typedef BCRSMatrix<FieldMatrix<double,1,1> > M;

  BCRSMatrix<FieldMatrix<double,1,1> > bcrsMatrix(3,3, BCRSMatrix<FieldMatrix<double,1,1> >::random);

  bcrsMatrix.setrowsize(0,1);
  bcrsMatrix.setrowsize(1,2);
  bcrsMatrix.setrowsize(2,2);

  bcrsMatrix.endrowsizes();

  bcrsMatrix.addindex(0, 0);
  bcrsMatrix.addindex(1, 1);
  bcrsMatrix.addindex(1, 0);
  bcrsMatrix.addindex(2, 2);
  bcrsMatrix.addindex(2, 1);
  bcrsMatrix.endindices();

  bcrsMatrix = 0;

  MatrixFunc mf;
  RowFunc rf;

  return testIterator<M,MatrixFunc,false>(bcrsMatrix,mf) +  testIterator<M::row_type,RowFunc,false>(bcrsMatrix[1], rf);
}
