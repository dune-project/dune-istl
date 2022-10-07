// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>

#include <dune/istl/bcrsmatrix.hh>

using namespace Dune;

int main (int argc, char** argv)
{
  typedef BCRSMatrix<FieldMatrix<double,2,2> >  Mat;

  Mat A(1,1, Mat::random);

  A.setrowsize(0,1);

  A.endrowsizes();

  A.addindex(0, 0);

  A.endindices();
  A = 0;

  Mat B(2,2, Mat::random);

  B.setrowsize(0,2);
  B.setrowsize(1,1);

  B.endrowsizes();

  B.addindex(0, 0);
  B.addindex(0, 1);

  B.addindex(1, 1);

  B.endindices();
  B = 0;

  B = A;

  return 0;
}
