#include"config.h"
#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/collectivecommunication.hh>
#include<dune/istl/paamg/twolevelmethod.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solvers.hh>

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
    typedef Dune::CollectiveCommunication<void*> Comm;
    Comm c;
    int n;
    BCRSMat mat = setupAnisotropic2d<BS,double>(N, indices, c, &n, 1);
    Vector b(mat.N()), x(mat.M());
    randomize(mat, b);
    typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> FSmoother;
    typedef Dune::SeqSOR<BCRSMat,Vector,Vector> CSmoother;
    typedef Dune::Amg::CoarsenCriterion<
        Dune::Amg::UnSymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
        Criterion;
    FSmoother fineSmoother(mat,1,1.0);
    typedef Dune::Amg::AggregationLevelTransferPolicy<Operator,Criterion>
        TransferPolicy;
    typedef Dune::Amg::OneStepAMGCoarseSolverPolicy<Operator,CSmoother, Criterion>
        CoarsePolicy;
    typedef typename Dune::Amg::SmootherTraits<CSmoother>::Arguments SmootherArgs;
    Criterion crit;
    CoarsePolicy coarsePolicy=CoarsePolicy(SmootherArgs(), crit);
    TransferPolicy transferPolicy(crit);
    Operator fop(mat);
    Dune::Amg::TwoLevelMethod<Operator,Operator,FSmoother> preconditioner(fop,
                                                                       Dune::stackobject_to_shared_ptr(fineSmoother),
                                                                       Dune::stackobject_to_shared_ptr<Dune::Amg::LevelTransferPolicy<Operator,Operator> >(transferPolicy),
                                                                       coarsePolicy);
    Dune::GeneralizedPCGSolver<Vector> amgCG(fop,preconditioner,1e-2,80,2);
    Dune::InverseOperatorResult res;
    amgCG.apply(x,b,res);
}

int main()
{
    testTwoLevelMethod();
    return 0;
}
