// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include<array>
#include<vector>

#include <dune/istl/cholmod.hh>


#include "laplacian.hh"

using namespace Dune;

// One ignore block
template<int k>
struct IgnoreBlock
{
  std::array<bool,k> data;

  bool operator[](std::size_t i) const
  {
    return data[i];

  }

  bool& operator[](std::size_t i)
  {
    return data[i];

  }

  std::size_t count() const
  {
    size_t res = 0;
    for(const auto& d : data)
      if (d)
        res++;

    return res;
  }
};

// Block ignore vector
template<int blocksize>
struct Ignore
{
  std::vector<IgnoreBlock<blocksize>> data;

  IgnoreBlock<blocksize> operator[](std::size_t i) const
  {
    return data[i];
  }

  IgnoreBlock<blocksize>& operator[](std::size_t i)
  {
    return data[i];
  }

  std::size_t count() const
  {
    size_t res = 0;
    for(const auto& d : data)
      res += d.count();

    return res;
  }
};


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
    Ignore<bs> ignore;
    ignore.data.resize(A.N());
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
