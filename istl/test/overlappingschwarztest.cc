// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/io.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <laplacian.hh>
#include <dune/common/timer.hh>
#include <dune/common/sllist.hh>
#include <dune/istl/overlappingschwarz.hh>
int main(int argc, char** argv)
{

  const int BS=1;
  int N=4;

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
  b=0;
  x=100;
  //setBoundary(x,b,N);
  /*
     for (BCRSMat::RowIterator row = mat.begin(); row != mat.end(); ++row)
     (*row)[row.index()]+=row.index();
   */
  // create the subdomains
  int domainSize=2;
  if(argc>2)
    domainSize = atoi(argv[2]);
  int overlap = 0;

  int domainsPerDim=(N+domainSize-1)/domainSize;

  // set up the overlapping domains
  typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector> Schwarz;
  typedef Schwarz::subdomain_vector subdomain_vector;

  subdomain_vector domains(domainsPerDim*domainsPerDim);


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

  typedef subdomain_vector::const_iterator iterator;

  if(N<10) {
    int i=0;
    for(iterator iter=domains.begin(); iter != domains.end(); ++iter) {
      typedef iterator::value_type::const_iterator entry_iterator;
      std::cout<<"domain "<<i++<<":";
      for(entry_iterator entry = iter->begin(); entry != iter->end(); ++entry) {
        std::cout<<" "<<*entry;
      }
      std::cout<<std::endl;
    }
    Dune::printmatrix(std::cout, mat, std::string("A"), std::string("A"));
    Dune::printvector(std::cout, b, std::string("B"), std::string("B"));
    Dune::printvector(std::cout, x, std::string("X"), std::string("X"));
  }

  Dune::SeqOverlappingSchwarz<BCRSMat,Vector> prec(mat, domains, 1);


  Dune::Timer watch;

  watch.reset();

  Dune::LoopSolver<Vector> solver(fop, prec, 1e-2,100,2);
  Dune::InverseOperatorResult res;


  //  b=0;
  //  x=100;
  //  setBoundary(x,b,N);
  std::cout<<"Additive Schwarz"<<std::endl;
  solver.apply(x,b, res);

  Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::MultiplicativeSchwarzMode> prec1(mat,domains, 1);
  Dune::LoopSolver<Vector> solver1(fop, prec1, 1e-2,100,2);

  b=0;
  x=100;
  //setBoundary(x,b,N);
  std::cout << "Multiplicative Schwarz"<<std::endl;

  solver1.apply(x,b, res);

  Dune::SeqSOR<BCRSMat,Vector,Vector> sor(mat, 1,1);
  Dune::LoopSolver<Vector> solver2(fop, sor, 1e-2,100,2);
  b=0;
  x=100;
  //setBoundary(x,b,N);
  std::cout << "SOR"<<std::endl;
  solver2.apply(x,b, res);
}
