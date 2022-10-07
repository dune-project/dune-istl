// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/common/tuplevector.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include<array>
#include<vector>

#include <dune/istl/cholmod.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>


#include "laplacian.hh"

using namespace Dune;

int main(int argc, char** argv)
{
#if HAVE_SUITESPARSE_UMFPACK
  try
  {

    int N = 30; // number of nodes
    const int bs = 2; // block size

    // fill matrix with external method
    BCRSMatrix<FieldMatrix<double,bs,bs>> A;
    setupLaplacian(A, N);

    BlockVector<FieldVector<double,bs>> b,x;
    b.resize(A.N());
    x.resize(A.N());
    b = 1;

    InverseOperatorResult res;

    // test without ignore nodes
    Cholmod<BlockVector<FieldVector<double,bs>>> cholmod;
    cholmod.setMatrix(A);
    cholmod.apply(x,b,res);

    // test
    A.mmv(x,b);

    if ( b.two_norm() > 1e-9 )
      std::cerr << " Error in CHOLMOD, residual is too large: " << b.two_norm() << std::endl;

    x = 0;
    b = 1;

    // test with ignore nodes
    std::vector<std::array<bool,bs>> ignore;
    ignore.resize(A.N());
    // ignore one random entry in x and b
    ignore[12][0] = true;
    b[12][0] = 666;
    x[12][0] = 123;


    Cholmod<BlockVector<FieldVector<double,bs>>> cholmod2;
    cholmod2.setMatrix(A,&ignore);
    cholmod2.apply(x,b,res);

    // check that x[12][0] is untouched
    if ( std::abs(x[12][0] - 123) > 1e-15 )
      std::cerr << " Error in CHOLMOD, x was NOT ignored!"<< std::endl;

    // reset the x value
    x[12][0] = 0;
    // test -> this should result in zero in every line except entry [12][1]
    A.mmv(x,b);
    auto b_12_0 = b[12][0];

    // check that error is completely caused by this entry
    if ( std::abs( b.two_norm() - std::abs(b_12_0) ) > 1e-15 )
      std::cerr << " Error in CHOLMOD, b was NOT ignored correctly: " << std::abs( b.two_norm() - std::abs(b_12_0) ) << std::endl;


    using BCRS = BCRSMatrix<FieldMatrix<double,bs,bs>>;
    using MTRow = MultiTypeBlockVector<BCRS,BCRS>;
    using MTBM = MultiTypeBlockMatrix<MTRow,MTRow>;

    MTBM A2;

    using namespace Indices;
    A2[_0][_0] = A;
    A2[_1][_0] = A;
    A2[_0][_1] = A;
    A2[_1][_1] = A;

    // this makes it regular
    A2[_0][_0] *= 2.;

    using MTBV = MultiTypeBlockVector<BlockVector<FieldVector<double,bs>>,BlockVector<FieldVector<double,bs>>>;
    MTBV b2,x2;

    b = 1;
    x = 0;


    x2[_0] = x;
    x2[_1] = x;

    b2[_0] = b;
    b2[_1] = b;

    Cholmod<MTBV> cholmodMT;
    cholmodMT.setMatrix(A2);

    cholmodMT.apply(x2,b2,res);

    // test
    A2.mmv(x2,b2);

    if ( b2.two_norm() > 1e-9 )
      std::cerr << " Error in CHOLMOD, residual is too large: " << b2.two_norm() << std::endl;


    x2 = 0;
    b2 = 1;

    // test with ignore nodes
    TupleVector<std::vector<std::array<bool,bs>>,std::vector<std::array<bool,bs>>> ignoreMT;
    ignoreMT[_0].resize(A.N());
    ignoreMT[_1].resize(A.N());
    // ignore one random entry in x and b
    ignoreMT[_0][12][0] = true;
    b2[_0][12][0] = 666;
    x2[_0][12][0] = 123;

    Cholmod<MTBV> cholmodMT2;
    cholmodMT2.setMatrix(A2,&ignoreMT);

    cholmodMT2.apply(x2,b2,res);

    // check that x was ignored
    if ( std::abs( x2[_0][12][0] - 123 ) > 1e-15 )
      std::cerr << " Error in CHOLMOD, x was NOT ignored correctly: " << std::abs( x2[_0][12][0] - 123 ) << std::endl;

    x2[_0][12][0] = 0;

    // test
    A2.mmv(x2,b2);

    auto b_0_12_0 = b2[_0][12][0];

    // check that error is completely caused by this entry
    if ( std::abs( b2.two_norm() - std::abs(b_0_12_0) ) > 1e-15 )
      std::cerr << " Error in CHOLMOD, b was NOT ignored correctly: " << std::abs( b2.two_norm() - std::abs(b_0_12_0) ) << std::endl;


  }
  catch (std::exception &e)
  {
    std::cout << "ERROR: " << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "Dune reported an unknown error." << std::endl;
    exit(1);
  }
#else // HAVE_SUITESPARSE_UMFPACK
  std::cerr << "You need SuiteSparse to run the CHOLMOD test." << std::endl;
  return 42;
#endif // HAVE_SUITESPARSE_UMFPACK
}
