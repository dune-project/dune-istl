// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include"config.h"
#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/istl/paamg/twolevelmethod.hh>
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solvers.hh>

#define NUM_THREADS 3

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

typedef double XREAL;
typedef Dune::ParallelIndexSet<int,LocalIndex,512> ParallelIndexSet;
typedef Dune::FieldMatrix<XREAL,1,1> MatrixBlock;
typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
typedef Dune::FieldVector<XREAL,1> VectorBlock;
typedef Dune::BlockVector<VectorBlock> Vector;
typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;
typedef Dune::Communication<void*> Comm;
typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> Smoother;
typedef Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
#ifndef USE_OVERLAPPINGSCHWARZ
typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> FSmoother;
#else
typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,
                                        Dune::SymmetricMultiplicativeSchwarzMode> FSmoother;
#endif
typedef Dune::SeqJac<BCRSMat,Vector,Vector> CSmoother;
typedef Dune::Amg::CoarsenCriterion<
        Dune::Amg::UnSymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
        Criterion;
typedef Dune::Amg::OneStepAMGCoarseSolverPolicy<Operator,CSmoother, Criterion>
        CoarsePolicy; // Policy for coarse solver creation
typedef Dune::Amg::TwoLevelMethod<Operator,CoarsePolicy,FSmoother> AMG;



struct thread_arg
{
  AMG *amg;
  Vector *b;
  Vector *x;
  Operator *fop;
};


void *solve(void* arg)
{
  thread_arg *amgarg=(thread_arg*) arg;

  Dune::GeneralizedPCGSolver<Vector> amgCG(*amgarg->fop,*amgarg->amg,1e-4,80,2);
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

void testTwoLevelMethod()
{
    const int BS=1;
    int N=100;
    typedef Dune::ParallelIndexSet<int,LocalIndex,512> ParallelIndexSet;
    ParallelIndexSet indices;
    typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
    typedef Dune::FieldVector<double,BS> VectorBlock;
    typedef Dune::BlockVector<VectorBlock> Vector;
    typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;
    typedef Dune::Communication<void*> Comm;
    Comm c;
    int n;
    BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, indices, c, &n, 1);
    Vector b(mat.N()), x(mat.M());
    x=0;
    randomize(mat, b);
#ifndef USE_OVERLAPPINGSCHWARZ
    // Smoother used for the finest level
    FSmoother fineSmoother(mat,1,1.0);
#else
    std::cout << "Use ovlps" << std::endl;
    // Smoother used for the finest level. There is an additional
    // template parameter to provide the subdomain solver.
    // Type of the subdomain vector
    typedef FSmoother::subdomain_vector SubdomainVector;
    // Create subdomain.
    std::size_t stride=2;
    SubdomainVector subdomains((((N-1)/stride)+1)*(((N-1)/stride)+1));

    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
        {
            int index=i/stride*(((N-1)/stride)+1)+j/stride;
            subdomains[index].insert(i*N+j);
        }
    //create smoother
    FSmoother fineSmoother(mat,subdomains, 1.0, false);
#endif
    // Create the approximate coarse level solver
    typedef Dune::SeqJac<BCRSMat,Vector,Vector> CSmoother;
    typedef Dune::Amg::CoarsenCriterion<
        Dune::Amg::UnSymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
        Criterion;
    typedef Dune::Amg::AggregationLevelTransferPolicy<Operator,Criterion>
        TransferPolicy; // Policy for coarse linear system creation
    typedef Dune::Amg::OneStepAMGCoarseSolverPolicy<Operator,CSmoother, Criterion>
        CoarsePolicy; // Policy for coarse solver creation
    typedef Dune::Amg::SmootherTraits<CSmoother>::Arguments SmootherArgs;
    Criterion crit;
    CoarsePolicy coarsePolicy=CoarsePolicy(SmootherArgs(), crit);
    TransferPolicy transferPolicy(crit);
    Operator fop(mat);
    Dune::Amg::TwoLevelMethod<Operator,CoarsePolicy,FSmoother> preconditioner(fop,
                                                                          Dune::stackobject_to_shared_ptr(fineSmoother),
                                                                          transferPolicy,
                                                                          coarsePolicy);
    Dune::GeneralizedPCGSolver<Vector> amgCG(fop,preconditioner,.8,80,2);
    Dune::Amg::TwoLevelMethod<Operator,CoarsePolicy,FSmoother> preconditioner1(preconditioner);
    Dune::InverseOperatorResult res;

  std::vector<AMG> amgs(NUM_THREADS, preconditioner1);
  std::vector<thread_arg> args(NUM_THREADS);
  std::vector<pthread_t> threads(NUM_THREADS);
    std::vector<Vector> xs(NUM_THREADS, x);
  std::vector<Vector> bs(NUM_THREADS, b);
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
  preconditioner.pre(x, b);
  amgs.resize(NUM_THREADS, preconditioner);
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

//    amgCG.apply(x,b,res);
}

int main()
{
    testTwoLevelMethod();
    return 0;
}
