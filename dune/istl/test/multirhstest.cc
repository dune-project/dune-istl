// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// start with including some headers
#include "config.h"

#include <iostream>               // for input/output to shell
#include <fstream>                // for input/output to files
#include <vector>                 // STL vector class
#include <complex>

#include <cmath>                 // Yes, we do some math here
#include <sys/times.h>            // for timing measurements

#include <dune/common/classname.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/simd.hh>
#include <dune/common/timer.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/basearray.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>

#include "laplacian.hh"

template<typename T>
struct Random {
  static T gen()
  {
    return drand48();
  }
};

template<typename T, typename A>
struct Random<Vc::Vector<T,A>> {
  static Vc::Vector<T,A> gen()
  {
    return Vc::Vector<T,A>::Random();
  }
};

template<typename T, std::size_t N, typename V, std::size_t M>
struct Random<Vc::SimdArray<T,N,V,M>> {
  static Vc::SimdArray<T,N,V,M> gen()
  {
    return Vc::SimdArray<T,N,V,M>::Random();
  }
};

template <typename V>
V detectVectorType(Dune::LinearOperator<V,V> &);

template<typename Operator, typename Solver>
void run_test (std::string precName, std::string solverName, Operator & op, Solver & solver, unsigned int N, unsigned int Runs)
{
  using Vector = decltype(detectVectorType(op));
  using FT = typename Vector::field_type;

  Dune::Timer t;
  for (unsigned int run = 0; run < Runs; run++) {
    // set up system
    Vector x(N),b(N);
    for (unsigned int i=0; i<N; i++)
      x[i] = Random<FT>::gen()/10.0;
    x=0; x[0]=1; x[N-1]=2; // prescribe known solution
    b=0; op.apply(x,b);    // set right hand side accordingly
    x=1;                   // initial guess

    // call the solver
    Dune::InverseOperatorResult r;
    solver.apply(x,b,r);
  }
  std::cout << "Test " << Runs << " run(s) " << solverName << "(" << precName << ")"
            << " with " << Dune::className<FT>() << " took " << t.stop() << std::endl;
}

template<typename Operator, typename Prec>
void test_all_solvers(std::string precName, Operator & op, Prec & prec, unsigned int N, unsigned int Runs)
{
  using Vector = decltype(detectVectorType(op));

  double reduction = 1e-4;
  int verb = 0;
  Dune::LoopSolver<Vector> loop(op,prec,reduction,18000,verb);
  Dune::CGSolver<Vector> cg(op,prec,reduction,8000,verb);
  Dune::BiCGSTABSolver<Vector> bcgs(op,prec,reduction,8000,verb);
  Dune::GradientSolver<Vector> grad(op,prec,reduction,18000,verb);
  // Dune::RestartedGMResSolver<Vector> gmres(op,prec,reduction,40,8000,verb);
  // Dune::MINRESSolver<Vector> minres(op,prec,reduction,8000,verb);
  Dune::GeneralizedPCGSolver<Vector> gpcg(op,prec,reduction,8000,verb);

  run_test(precName, "Loop",           op,loop,N,Runs);
  run_test(precName, "CG",             op,cg,N,Runs);
  run_test(precName, "Gradient",       op,bcgs,N,Runs);
  run_test(precName, "RestartedGMRes", op,grad,N,Runs);
  // run_test(precName,                   op,gmres,N,Runs);
  // run_test(precName, "MINRes",         op,minres,N,Runs);
  run_test(precName, "GeneralizedPCG", op,gpcg,N,Runs);
}

template<typename FT>
void test_all(unsigned int Runs = 1)
{
  // define Types
  typedef Dune::FieldVector<FT,1> VB;
  typedef Dune::FieldMatrix<double,1,1> MB;
  typedef Dune::BlockVector<VB> Vector;
  typedef Dune::BCRSMatrix<MB> Matrix;

  // size
  unsigned int size = 100;
  unsigned int N = size*size;

  // make a compressed row matrix with five point stencil
  Matrix A;
  setupLaplacian(A,size);
  Dune::MatrixAdapter<Matrix,Vector,Vector> op(A);        // make linear operator from A

  // create all preconditioners
  Dune::SeqJac<Matrix,Vector,Vector> jac(A,1,1);          // Jacobi preconditioner
  Dune::SeqGS<Matrix,Vector,Vector> gs(A,1,0.5);          // GS preconditioner
  Dune::SeqSOR<Matrix,Vector,Vector> sor(A,1,1.9520932);  // SOR preconditioner
  Dune::SeqSSOR<Matrix,Vector,Vector> ssor(A,1,1.0);      // SSOR preconditioner
  Dune::SeqILU0<Matrix,Vector,Vector> ilu0(A,1.0);        // preconditioner object
  Dune::SeqILUn<Matrix,Vector,Vector> ilu1(A,1,0.92);     // preconditioner object

  // run the sub-tests
  test_all_solvers("Jacobi",      op,jac,N,Runs);
  test_all_solvers("GaussSeidel", op,gs,N,Runs);
  test_all_solvers("SOR",         op,sor,N,Runs);
  test_all_solvers("SSOR",        op,ssor,N,Runs);
  test_all_solvers("ILU0",        op,ilu0,N,Runs);
  test_all_solvers("ILU1",        op,ilu1,N,Runs);
}

int main ()
{
  test_all<float>();
  test_all<double>();
  test_all<Vc::double_v>();
#if HAVE_VC
  test_all<Vc::Vector<double, Vc::VectorAbi::Scalar>>();
  test_all<Vc::SimdArray<double,2>>();
  test_all<Vc::SimdArray<double,2,Vc::Vector<double, Vc::VectorAbi::Scalar>,1>>();
  test_all<Vc::SimdArray<double,8>>();
  test_all<Vc::SimdArray<double,8,Vc::Vector<double, Vc::VectorAbi::Scalar>,1>>();
#endif
  test_all<double>(8);

  return 0;
}
