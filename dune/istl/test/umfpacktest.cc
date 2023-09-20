// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/timer.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/umfpack.hh>

#include "laplacian.hh"

using namespace Dune;

template<typename Matrix, typename Vector, typename BitVector>
TestSuite runUMFPack(std::size_t N)
{
  TestSuite t;

  Matrix mat;
  Vector b(N*N), x(N*N), b1(N/2), x1(N/2);

  // with additional diagonal regularization
  setupLaplacian(mat,N,100.0);

  b=1;
  b1=1;
  x=0;
  x1=0;

  Dune::Timer watch;

  watch.reset();

  Dune::UMFPack<Matrix> solver(mat,1);

  Dune::InverseOperatorResult res;

  solver.apply(x, b, res);
  solver.free();

  // test
  auto norm = b.two_norm();
  mat.mmv(x,b);
  t.check( b.two_norm()/norm < 1e-11 ) << " Error in UMFPACK, relative residual is too large: " << b.two_norm()/norm;

  Dune::UMFPack<Matrix> solver1;

  std::set<std::size_t> mrs;
  BitVector bitVector(N*N);
  bitVector = true;
  for(std::size_t s=0; s < N/2; ++s)
  {
    mrs.insert(s);
    bitVector[s] = false;
  }

  solver1.setMatrix(mat,bitVector);
  solver1.apply(x1,b1, res);

  // test
  x=0;
  b=0;
  for( std::size_t i=0; i<N/2; i++ )
  {
    // set subindices
    x[i] = x1[i];
    b[i] = b1[i];
  }

  norm = b.two_norm();
  mat.mmv(x,b);

  // truncate deactivated indices
  for( std::size_t i=N/2; i<N*N; i++ )
    b[i] = 0;

  t.check( b.two_norm()/norm < 1e-11 ) << " Error in UMFPACK, relative residual is too large: " << b.two_norm()/norm;

  // compare with setSubMatrix
  solver1.setSubMatrix(mat,mrs);
  solver1.setVerbosity(true);

  auto x2 = x1;
  x2 = 0;
  solver1.apply(x2,b1, res);

  x2 -= x1;
  t.check( x2.two_norm()/x1.two_norm() < 1e-11 ) << " Error in UMFPACK, setSubMatrix yields different result as setMatrix with BitVector, relative diff: " << x2.two_norm()/x1.two_norm();


  solver1.apply(reinterpret_cast<typename Matrix::field_type*>(&x1[0]),
                reinterpret_cast<typename Matrix::field_type*>(&b1[0]));

  Dune::UMFPack<Matrix> save_solver(mat,"umfpack_decomp",0);
  Dune::UMFPack<Matrix> load_solver(mat,"umfpack_decomp",0);

  return t;
}



template<typename Matrix, typename Vector, typename BitVector>
TestSuite runUMFPackMultitype2x2(std::size_t N)
{
  TestSuite t;

  using namespace Dune::Indices;

  Matrix mat;

  Vector b,x,b1,x1;

  b[_0].resize(N*N);
  b[_1].resize(N*N);

  x[_0].resize(N*N);
  x[_1].resize(N*N);

  b1[_0].resize(N/2);
  b1[_1].resize(N/2);
  x1[_0].resize(N/2);
  x1[_1].resize(N/2);


  // initialize all 4 matrix blocks
  setupLaplacian(mat[_0][_0],N,100.0); // regularize the diagonal block
  setupLaplacian(mat[_0][_1],N);
  setupLaplacian(mat[_1][_0],N);
  setupLaplacian(mat[_1][_1],N,200.0); // regularize the diagonal block

  b=1.0;
  b1=1.0;
  x=0.0;
  x1=0.0;

  Dune::Timer watch;

  watch.reset();

  Dune::UMFPack<Matrix> solver(mat,1);

  Dune::InverseOperatorResult res;

  solver.apply(x, b, res);
  solver.free();

  // test
  auto norm = b.two_norm();
  mat.mmv(x,b);
  t.check( b.two_norm()/norm < 1e-11 ) << " Error in UMFPACK, relative residual is too large: " << b.two_norm()/norm;

  Dune::UMFPack<Matrix> solver1;

  BitVector bitVector;
  bitVector[_0].resize(N*N);
  bitVector[_1].resize(N*N);

  bitVector = true;
  for(std::size_t s=0; s < N/2; ++s)
  {
    bitVector[_0][s] = false;
    bitVector[_1][s] = false;
  }

  solver1.setMatrix(mat,bitVector);
  solver1.apply(x1,b1, res);

  // test
  x=0;
  b=0;
  for( std::size_t i=0; i<N/2; i++ )
  {
    // set subindices
    x[_0][i] = x1[_0][i];
    x[_1][i] = x1[_1][i];
    b[_0][i] = b1[_0][i];
    b[_1][i] = b1[_1][i];
  }

  norm = b.two_norm();
  mat.mmv(x,b);

  // truncate deactivated indices
  for( std::size_t i=N/2; i<N*N; i++ )
  {
    b[_0][i] = 0;
    b[_1][i] = 0;
  }

  t.check( b.two_norm()/norm < 1e-11 ) << " Error in UMFPACK, relative residual is too large: " << b.two_norm()/norm;

  return t;
}


int main(int argc, char** argv) try
{
#if HAVE_SUITESPARSE_UMFPACK

  TestSuite t;
  std::size_t N=100;

  if(argc>1)
    N = atoi(argv[1]);

  // ------------------------------------------------------------------------------
  std::cout<<"testing for N="<<N<<" BCRSMatrix<double>"<<std::endl;
  {
    using Matrix    = Dune::BCRSMatrix<double>;
    using Vector    = Dune::BlockVector<double>;
    using BitVector = Dune::BlockVector<int>;
    t.subTest(runUMFPack<Matrix,Vector,BitVector>(N));
  }

  // ------------------------------------------------------------------------------
  std::cout<<"testing for N="<<N<<" BCRSMatrix<FieldMatrix<double,1,1> >"<<std::endl;
  {
    using Matrix    = Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> >;
    using Vector    = Dune::BlockVector<Dune::FieldVector<double,1> >;
    using BitVector = Dune::BlockVector<Dune::FieldVector<int,1> >;
    t.subTest(runUMFPack<Matrix,Vector,BitVector>(N));
  }

  // ------------------------------------------------------------------------------
  std::cout<<"testing for N="<<N<<" BCRSMatrix<FieldMatrix<double,2,2> >"<<std::endl;
  {
    using Matrix    = Dune::BCRSMatrix<Dune::FieldMatrix<double,2,2> >;
    using Vector    = Dune::BlockVector<Dune::FieldVector<double,2> >;
    using BitVector = Dune::BlockVector<Dune::FieldVector<int,2> >;
    t.subTest(runUMFPack<Matrix,Vector,BitVector>(N));
  }

  // ------------------------------------------------------------------------------
  std::cout<<"testing for N="<<N<<" MultiTypeBlockMatrix<BCRSMatrix<FieldMatrix<double,2,2>> ... >"<<std::endl;
  {
    using MatrixBlock    = Dune::BCRSMatrix<Dune::FieldMatrix<double,2,2> >;
    using BlockRow       = Dune::MultiTypeBlockVector<MatrixBlock,MatrixBlock>;
    using Matrix         = Dune::MultiTypeBlockMatrix<BlockRow,BlockRow>;

    using VectorBlock    = Dune::BlockVector<Dune::FieldVector<double,2> >;
    using Vector         = Dune::MultiTypeBlockVector<VectorBlock,VectorBlock>;

    using BitVectorBlock = Dune::BlockVector<Dune::FieldVector<int,2> >;
    using BitVector      = Dune::MultiTypeBlockVector<BitVectorBlock,BitVectorBlock>;

    t.subTest(runUMFPackMultitype2x2<Matrix,Vector,BitVector>(N));
  }



  return t.exit();
#else // HAVE_SUITESPARSE_UMFPACK
  std::cerr << "You need SuiteSparse's UMFPack to run this test." << std::endl;
  return 77;
#endif // HAVE_SUITESPARSE_UMFPACK
}
catch (std::exception &e)
{
  std::cout << "ERROR: " << e.what() << std::endl;
  return 1;
}
