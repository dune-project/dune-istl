#include <config.h>
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/cholmod.hh>


#include "laplacian.hh"

using namespace Dune;

// dummy for empty ignore block
struct IgnoreBlock
{
  bool operator[](std::size_t) const {return false;}
};

// dummy for empty ignore block vector
struct Ignore
{
  IgnoreBlock operator[](std::size_t) const { return IgnoreBlock(); }
  std::size_t count() const { return 0; }
};


int main(int argc, char** argv)
{
#if HAVE_SUITESPARSE_UMFPACK
  try
  {

    int N = 300; // number of nodes
    const int bs = 1; // block size

    // fill matrix with external method
    BCRSMatrix<FieldMatrix<double,bs,bs>> A;
    setupLaplacian(A, N);

    BlockVector<FieldVector<double,bs>> b,x;
    b.resize(A.N());
    b = 1;

    InverseOperatorResult res;

  // test without ignore nodes
    Cholmod<BCRSMatrix<FieldMatrix<double,bs,bs>>> cholmod;
    cholmod.setMatrix(A);
    cholmod.apply(x,b,res);

    // test
    A.mmv(x,b);

    if ( b.two_norm() > 1e-9 )
      std::cerr << " Error in CHOLMOD, residual is too large: " << b.two_norm() <<  "\n";

    x = 0;
    b = 1;

  // test with ignore nodes
    Ignore ignore;
    Cholmod<BCRSMatrix<FieldMatrix<double,bs,bs>>> cholmod2;
    cholmod2.setMatrix(A,&ignore);
    cholmod2.apply(x,b,res);

    // test
    A.mmv(x,b);

    if ( b.two_norm() > 1e-9 )
      std::cerr << " Error in CHOLMOD, residual is too large: " << b.two_norm() <<  "\n";

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
