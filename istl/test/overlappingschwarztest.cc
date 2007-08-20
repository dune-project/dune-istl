// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <laplacian.hh>
#include <dune/common/timer.hh>
#include <dune/istl/overlappingschwarz.hh>
int main(int argc, char** argv)
{

  const int BS=1;
  int N=100;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;


  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  Vector b(N*N), x(N*N);

  setupLaplacian(mat,N);
  b=1;
  x=0;

  // create the subdomains
  int domainSize=4;
  int overlap = 1;

  int domainsPerDim=(N+domainSize-1)/domainSize;

  // set up the overlapping domains
  std::vector<std::set<BCRSMat::size_type> > domains(domainsPerDim*domainsPerDim);


  for(int j=0; j < N; ++j)
    for(int i=0; i < N; ++i)
    {
      int xdomain = i/domainSize;
      int ydomain = j/domainSize;
      int mainDomain=ydomain*domainsPerDim+xdomain;
      domains[mainDomain].insert(j*N+i);

      // check left domain
      int domain = (i-overlap)/domainSize;
      if(domain>=0 && domain<domainsPerDim)
        domains[ydomain*domainsPerDim+domain].insert(j*N+i);

      //check right domain
      domain = (i+overlap)/domainSize;
      if(domain>=0 && domain<domainsPerDim)
        domains[ydomain*domainsPerDim+domain].insert(j*N+i);

      // check lower domain
      domain = (j-overlap)/domainSize;
      if(domain>=0 && domain<domainsPerDim)
        domains[domain*domainsPerDim+xdomain].insert(j*N+i);

      //check right domain
      domain = (j+overlap)/domainSize;
      if(domain>=0 && domain<domainsPerDim)
        domains[domain*domainsPerDim+xdomain].insert(j*N+i);
    }

  typedef std::vector<std::set<BCRSMat::size_type> >::const_iterator iterator;
  int i=0;
  for(iterator iter=domains.begin(); iter != domains.end(); ++iter) {
    typedef std::set<BCRSMat::size_type>::const_iterator entry_iterator;
    std::cout<<"domain "<<i++<<":";
    for(entry_iterator entry = iter->begin(); entry != iter->end(); ++entry) {
      std::cout<<" "<<*entry;
    }
    std::cout<<std::endl;
  }

  Dune::SeqOverlappingSchwarz<BCRSMat,Vector> prec(mat, domains);


  Dune::Timer watch;

  watch.reset();

  Dune::LoopSolver<Vector> solver(fop, prec, 10e-8,80,2);
  Dune::InverseOperatorResult res;

  solver.apply(x,b, res);

}
