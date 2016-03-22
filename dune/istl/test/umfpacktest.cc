#include <config.h>

#include <complex>
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/colcompmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/umfpack.hh>

#include "laplacian.hh"


int main(int argc, char** argv)
{
#if HAVE_SUITESPARSE_UMFPACK
  try
  {
    typedef double FIELD_TYPE;
    //typedef std::complex<double> FIELD_TYPE;

    const int BS=1;
    std::size_t N=100;

    if(argc>1)
      N = atoi(argv[1]);
    std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;

    typedef Dune::FieldMatrix<FIELD_TYPE,BS,BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
    typedef Dune::FieldVector<FIELD_TYPE,BS> VectorBlock;
    typedef Dune::BlockVector<VectorBlock> Vector;
    typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;

    BCRSMat mat;
    Operator fop(mat);
    Vector b(N*N), x(N*N), b1(N/2), x1(N/2);

    setupLaplacian(mat,N);
    b=1;
    b1=1;
    x=0;
    x1=0;

    Dune::Timer watch;

    watch.reset();

    Dune::UMFPack<BCRSMat> solver(mat,1);

    Dune::InverseOperatorResult res;

    solver.apply(x, b, res);
    solver.free();

    Dune::UMFPack<BCRSMat> solver1;

    std::set<std::size_t> mrs;
    for(std::size_t s=0; s < N/2; ++s)
      mrs.insert(s);

    solver1.setSubMatrix(mat,mrs);
    solver1.setVerbosity(true);

    solver1.apply(x1,b1, res);
    solver1.apply(reinterpret_cast<FIELD_TYPE*>(&x1[0]), reinterpret_cast<FIELD_TYPE*>(&b1[0]));

    Dune::UMFPack<BCRSMat> save_solver(mat,"umfpack_decomp",0);
    Dune::UMFPack<BCRSMat> load_solver(mat,"umfpack_decomp",0);
    return 0;
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
  std::cerr << "You need SuiteSparse's UMFPack to run this test." << std::endl;
  return 77;
#endif // HAVE_SUITESPARSE_UMFPACK
}
