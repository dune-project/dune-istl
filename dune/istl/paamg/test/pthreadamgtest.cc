// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

/*#include"xreal.h"

   namespace std
   {

   HPA::xreal abs(const HPA::xreal& t)
   {
    return t>=0?t:-t;
   }

   };
 */

#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/fastamg.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solvers.hh>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#define NUM_THREADS 3

typedef double XREAL;

/*
   typedef HPA::xreal XREAL;

   namespace Dune
   {
   template<>
   struct DoubleConverter<HPA::xreal>
   {
   static double toDouble(const HPA::xreal& t)
   {
     return t._2double();
   }
   };
   }
 */

template<class M, class V>
void randomize(const M& mat, V& b)
{
  V x=b;

  srand((unsigned)std::clock());

  typedef typename V::iterator iterator;
  for(iterator i=x.begin(); i != x.end(); ++i)
    *i=(rand() / (RAND_MAX + 1.0));

  mat.mv(static_cast<const V&>(x), b);
}

typedef Dune::ParallelIndexSet<int,LocalIndex,512> ParallelIndexSet;
typedef Dune::FieldMatrix<XREAL,1,1> MatrixBlock;
typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
typedef Dune::FieldVector<XREAL,1> VectorBlock;
typedef Dune::BlockVector<VectorBlock> Vector;
typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;
typedef Dune::Communication<void*> Comm;
typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> Smoother;
typedef Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;

struct thread_arg
{
  MYAMG *amg;
  Vector *b;
  Vector *x;
  Operator *fop;
};


void *solve(void* arg)
{
  thread_arg *amgarg=(thread_arg*) arg;

  Dune::GeneralizedPCGSolver<Vector> amgCG(*amgarg->fop,*amgarg->amg,1e-6,80,2);
  //Dune::LoopSolver<Vector> amgCG(fop, amg, 1e-4, 10000, 2);
  Dune::Timer watch;
  Dune::InverseOperatorResult r;
  amgCG.apply(*amgarg->x,*amgarg->b,r);

  double solvetime = watch.elapsed();

  std::cout<<"AMG solving took "<<solvetime<<" seconds"<<std::endl;

  pthread_exit(NULL);
}

void *solve2(void* arg)
{
  thread_arg *amgarg=(thread_arg*) arg;
  *amgarg->x=0;
  (*amgarg->amg).pre(*amgarg->x,*amgarg->b);
  (*amgarg->amg).apply(*amgarg->x,*amgarg->b);
  (*amgarg->amg).post(*amgarg->x);
  return 0;
}

template <int BS, typename AMG>
void testAMG(int N, int coarsenTarget, int ml)
{

  std::cout<<"N="<<N<<" coarsenTarget="<<coarsenTarget<<" maxlevel="<<ml<<std::endl;



  ParallelIndexSet indices;
  int n;

  Comm c;
  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, indices, c, &n, 1);

  Vector b(mat.N()), x(mat.M());

  b=0;
  x=100;

  setBoundary(x, b, N);

  x=0;
  randomize(mat, b);

  std::vector<Vector> xs(NUM_THREADS, x);
  std::vector<Vector> bs(NUM_THREADS, b);

  if(N<6) {
    Dune::printmatrix(std::cout, mat, "A", "row");
    Dune::printvector(std::cout, x, "x", "row");
  }

  Dune::Timer watch;

  watch.reset();
  Operator fop(mat);

  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
  Criterion;
  //typedef Dune::SeqSOR<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqJac<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::MultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::SymmetricMultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector> Smoother;

  SmootherArgs smootherArgs;

  smootherArgs.iterations = 1;

  //smootherArgs.overlap=SmootherArgs::vertex;
  //smootherArgs.overlap=SmootherArgs::none;
  //smootherArgs.overlap=SmootherArgs::aggregate;

  smootherArgs.relaxationFactor = 1;

  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setMaxLevel(ml);
  criterion.setSkipIsolated(false);

  Dune::SeqScalarProduct<Vector> sp;

  Smoother smoother(mat,1,1);
  AMG amg(fop, criterion);


  double buildtime = watch.elapsed();

  std::cout<<"Building hierarchy took "<<buildtime<<" seconds"<<std::endl;
  std::vector<AMG> amgs(NUM_THREADS, amg);
  std::vector<thread_arg> args(NUM_THREADS);
  std::vector<pthread_t> threads(NUM_THREADS);
  for(int i=0; i < NUM_THREADS; ++i)
  {
    args[i].amg=&amgs[i];
    args[i].b=&bs[i];
    args[i].x=&xs[i];
    args[i].fop=&fop;
    pthread_create(&threads[i], NULL, solve, (void*) &args[i]);
  }
  void* retval;

  for(int i=0; i < NUM_THREADS; ++i)
    pthread_join(threads[i], &retval);

  amgs.clear();
  args.clear();
  amg.pre(x, b);
  amgs.resize(NUM_THREADS, amg);
  for(int i=0; i < NUM_THREADS; ++i)
  {
    args[i].amg=&amgs[i];
    args[i].b=&bs[i];
    args[i].x=&xs[i];
    args[i].fop=&fop;
    pthread_create(&threads[i], NULL, solve2, (void*) &args[i]);
  }
  for(int i=0; i < NUM_THREADS; ++i)
    pthread_join(threads[i], &retval);

}


int main(int argc, char** argv)
try
{
  int N=100;
  int coarsenTarget=1200;
  int ml=10;

  if(argc>1)
    N = atoi(argv[1]);

  if(argc>2)
    coarsenTarget = atoi(argv[2]);

  if(argc>3)
    ml = atoi(argv[3]);

  testAMG<1,MYAMG>(N, coarsenTarget, ml);

  //testAMG<2>(N, coarsenTarget, ml);

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
