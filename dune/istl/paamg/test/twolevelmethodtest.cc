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
    typedef Dune::Communication<void*> Comm;
    Comm c;
    int n;
    BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, indices, c, &n, 1);
    Vector b(mat.N()), x(mat.M());
    x=0;
    randomize(mat, b);
#ifndef USE_OVERLAPPINGSCHWARZ
    // Smoother used for the finest level
    typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> FSmoother;
    FSmoother fineSmoother(mat,1,1.0);
#else
    // Smoother used for the finest level. There is an additional
    // template parameter to provide the subdomain solver.
    typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,
                                        Dune::SymmetricMultiplicativeSchwarzMode> FSmoother;
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
    typedef Dune::SeqJac<BCRSMat,Vector,Vector> CSmoother;
    typedef Dune::Amg::SmootherTraits<CSmoother>::Arguments SmootherArgs;
    typedef Dune::Amg::CoarsenCriterion<
      Dune::Amg::UnSymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> > Criterion;
    typedef Dune::Amg::AggregationLevelTransferPolicy<Operator,Criterion> TransferPolicy;
    typedef Dune::Amg::OneStepAMGCoarseSolverPolicy<Operator,CSmoother, Criterion> CoarsePolicy; // Policy for coarse solver creation
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
    amgCG.apply(x,b,res);
}

int main()
{
    testTwoLevelMethod();
    return 0;
}
