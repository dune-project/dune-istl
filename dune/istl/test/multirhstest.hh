// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_TEST_MULTIRHSTEST_HH
#define DUNE_ISTL_TEST_MULTIRHSTEST_HH

#define DISABLE_AMG_DIRECTSOLVER 1

#include <iostream>               // for input/output to shell
#include <fstream>                // for input/output to files
#include <vector>                 // STL vector class
#include <complex>

#include <cmath>                 // Yes, we do some math here
#include <sys/times.h>            // for timing measurements

#include <dune/common/alignedallocator.hh>
#include <dune/common/classname.hh>
#include <dune/common/debugalign.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#if HAVE_VC
#include <dune/common/simd/vc.hh>
#endif
#include <dune/common/timer.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/basearray.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <dune/istl/test/laplacian.hh>

template<typename T>
struct Random {
  static T gen()
  {
    return drand48();
  }
};

#if HAVE_VC
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
#endif

template <typename V>
V detectVectorType(Dune::LinearOperator<V,V> &);

template<typename Operator, typename Solver>
std::vector<double> run_test (std::string precName, std::string solverName, Operator & op, Solver & solver, unsigned int N, unsigned int Runs)
{
  using Vector = decltype(detectVectorType(op));
  using FT = typename Vector::field_type;

  Dune::Timer t(false);
  std::vector<double> timestamps;

  std::cout << "Trying " << solverName << "(" << precName << ")"
            << " with " << Dune::className<FT>() << std::endl;
  for (unsigned int run = 0; run < Runs; run++) {
    // set up system
    Vector x(N),b(N);
    for (unsigned int i=0; i<N; i++)
      x[i] = Random<FT>::gen();
    b=0; op.apply(x,b);    // set right hand side accordingly
    x=1;                   // initial guess

    // call the solver
    Dune::InverseOperatorResult r;
    t.start();
    solver.apply(x,b,r);
    t.stop();
    double time = t.lastElapsed();
    timestamps.push_back(time);
  }

  double measuredTime = 0.0;
  for(auto d : timestamps)
    measuredTime += d;

  std::cout << Runs << " run(s) took " << measuredTime << std::endl;

  return timestamps;
}

template<typename Operator, typename Prec>
void test_all_solvers(std::string precName, Operator & op, Prec & prec, unsigned int N, unsigned int Runs)
{
  using Vector = decltype(detectVectorType(op));

  double reduction = 1e-1;
  int verb = 1;
  Dune::LoopSolver<Vector> loop(op,prec,reduction,18000,verb);
  Dune::CGSolver<Vector> cg(op,prec,reduction,8000,verb);
  Dune::BiCGSTABSolver<Vector> bcgs(op,prec,reduction,8000,verb);
  Dune::GradientSolver<Vector> grad(op,prec,reduction,18000,verb);
  Dune::RestartedGMResSolver<Vector> gmres(op,prec,reduction,40,8000,verb);
  Dune::MINRESSolver<Vector> minres(op,prec,reduction,8000,verb);
  Dune::GeneralizedPCGSolver<Vector> gpcg(op,prec,reduction,8000,verb);
  Dune::RestartedFCGSolver<Vector> rfcg(op,prec,reduction,8000,verb);
  Dune::CompleteFCGSolver<Vector> cfcg(op,prec,reduction,8000,verb);

  // run_test(precName, "Loop",           op,loop,N,Runs);
  run_test(precName, "CG",             op,cg,N,Runs);
  run_test(precName, "BiCGStab",       op,bcgs,N,Runs);
  run_test(precName, "Gradient",       op,grad,N,Runs);
  run_test(precName, "RestartedGMRes", op,gmres,N,Runs);
  run_test(precName, "MINRes",         op,minres,N,Runs);
  run_test(precName, "GeneralizedPCG", op,gpcg,N,Runs);
  run_test(precName, "RestartedFCG",   op,rfcg,N,Runs);
  run_test(precName, "CompleteFCG",    op,cfcg,N,Runs);
}

template<typename FT>
void test_all(unsigned int Runs = 1)
{
  // define Types
  typedef typename Dune::Simd::Scalar<FT> MT;
  typedef Dune::FieldVector<FT,1> VB;
  typedef Dune::FieldMatrix<MT,1,1> MB;
  typedef Dune::AlignedAllocator<VB> AllocV;
  typedef Dune::BlockVector<VB,AllocV> Vector;
  typedef Dune::BCRSMatrix<MB> Matrix;

  // size
  unsigned int size = 100;
  unsigned int N = size*size;

  // make a compressed row matrix with five point stencil
  Matrix A;
  setupLaplacian(A,size);
  typedef Dune::MatrixAdapter<Matrix,Vector,Vector> Operator;
  Operator op(A);        // make linear operator from A

  // create all preconditioners
  Dune::SeqJac<Matrix,Vector,Vector> jac(A,1,0.1);          // Jacobi preconditioner
  Dune::SeqGS<Matrix,Vector,Vector> gs(A,1,0.1);          // GS preconditioner
  Dune::SeqSOR<Matrix,Vector,Vector> sor(A,1,0.1);  // SOR preconditioner
  Dune::SeqSSOR<Matrix,Vector,Vector> ssor(A,1,0.1);      // SSOR preconditioner
  Dune::SeqILU<Matrix,Vector,Vector> ilu_0(A,0.1);       // preconditioner object
  Dune::SeqILU<Matrix,Vector,Vector> ilu_1(A,1,0.1);     // preconditioner object

  // AMG
  typedef Dune::Amg::RowSum Norm;
  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Matrix,Norm> >
          Criterion;
  typedef Dune::SeqSSOR<Matrix,Vector,Vector> Smoother;
  typedef typename Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
  SmootherArgs smootherArgs;
  smootherArgs.iterations = 1;
  smootherArgs.relaxationFactor = 1;
  unsigned int coarsenTarget = 1000;
  unsigned int maxLevel = 10;
  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setMaxLevel(maxLevel);
  criterion.setSkipIsolated(false);
  criterion.setNoPreSmoothSteps(1);
  criterion.setNoPostSmoothSteps(1);
  Dune::SeqScalarProduct<Vector> sp;
  typedef Dune::Amg::AMG<Operator,Vector,Smoother,Dune::Amg::SequentialInformation> AMG;
  Smoother smoother(A,1,1);
  AMG amg(op, criterion, smootherArgs);

  // run the sub-tests
  test_all_solvers("Jacobi",      op,jac,N,Runs);
  test_all_solvers("GaussSeidel", op,gs,N,Runs);
  test_all_solvers("SOR",         op,sor,N,Runs);
  test_all_solvers("SSOR",        op,ssor,N,Runs);
  test_all_solvers("ILU(0)",      op,ilu_0,N,Runs);
  test_all_solvers("ILU(1)",      op,ilu_1,N,Runs);
  test_all_solvers("AMG",         op,amg,N,Runs);
}

#endif // DUNE_ISTL_TEST_MULTIRHSTEST_HH
