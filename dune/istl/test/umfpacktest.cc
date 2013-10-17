#include "config.h"

#include <complex>
#include<iostream>

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
    Vector b(N*N), x(N*N);

    setupLaplacian(mat,N);
    b=1;
    x=0;

    Dune::Timer watch;

    watch.reset();

    Dune::UMFPack<BCRSMat> solver(mat,1);

    Dune::InverseOperatorResult res;

    Dune::UMFPack<BCRSMat> solver1;

    std::set<std::size_t> mrs;
    for(std::size_t s=0; s < N/2; ++s)
      mrs.insert(s);

    solver1.setSubMatrix(mat,mrs);
    solver1.setVerbosity(true);

    solver.apply(x,b, res);
    solver.free();

    solver1.apply(x,b, res);
    solver1.apply(reinterpret_cast<FIELD_TYPE*>(&x[0]), reinterpret_cast<FIELD_TYPE*>(&b[0]));

    Dune::UMFPack<BCRSMat> save_solver(mat,"umfpack_decomp",0);
    Dune::UMFPack<BCRSMat> load_solver(mat,"umfpack_decomp",0);
    return 0;
  }
  catch(Dune::Exception &e)
  {
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...)
  {}
}
