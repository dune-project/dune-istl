// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include <config.h>

#include <dune/istl/io.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/test/testsuite.hh>
#include "laplacian.hh"
#include <dune/common/timer.hh>
#include <dune/common/sllist.hh>
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/solvers.hh>
#include<dune/istl/superlu.hh>
#include<dune/istl/umfpack.hh>

#include <iterator>

template<class MatrixBlock, class VectorBlock>
Dune::TestSuite test(int argc, char** argv)
{
  Dune::TestSuite suite;
#if HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK
  int N=4;

  if(argc>1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<std::endl;


  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::BlockVector<VectorBlock> BVector;
  typedef Dune::MatrixAdapter<BCRSMat,BVector,BVector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  BVector b(N*N), x(N*N);

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
  typedef Dune::SeqOverlappingSchwarz<BCRSMat,BVector> Schwarz;
  typedef typename Schwarz::subdomain_vector subdomain_vector;

  subdomain_vector domains(domainsPerDim*domainsPerDim);

  // set up the rowToDomain vector
  typedef typename Schwarz::rowtodomain_vector rowtodomain_vector;
  rowtodomain_vector rowToDomain(N*N);

  for(int j=0; j < N; ++j)
    for(int i=0; i < N; ++i)
    {
      int xdomain = i/domainSize;
      int ydomain = j/domainSize;
      int mainDomain=ydomain*domainsPerDim+xdomain;
      int id=j*N+i;
      domains[mainDomain].insert(id);
      rowToDomain[id].push_back(mainDomain);

      // check left domain
      int domain = (i-overlap)/domainSize;
      int neighbourDomain=ydomain*domainsPerDim+domain;
      if(domain>=0 && domain<domainsPerDim && neighbourDomain!=mainDomain)
      {
        domains[neighbourDomain].insert(id);
        rowToDomain[id].push_back(neighbourDomain);
      }

      //check right domain
      domain = (i+overlap)/domainSize;
      neighbourDomain=ydomain*domainsPerDim+domain;
      if(domain>=0 && domain<domainsPerDim && neighbourDomain!=mainDomain)
      {
        domains[neighbourDomain].insert(id);
        rowToDomain[id].push_back(neighbourDomain);
      }

      // check lower domain
      domain = (j-overlap)/domainSize;
      neighbourDomain=domain*domainsPerDim+xdomain;
      if(domain>=0 && domain<domainsPerDim && neighbourDomain!=mainDomain)
      {
        domains[neighbourDomain].insert(id);
        rowToDomain[id].push_back(neighbourDomain);
      }

      //check upper domain
      domain = (j+overlap)/domainSize;
      neighbourDomain=domain*domainsPerDim+xdomain;
      if(domain>=0 && domain<domainsPerDim && neighbourDomain!=mainDomain)
      {
        domains[neighbourDomain].insert(id);
        rowToDomain[id].push_back(domain*domainsPerDim+xdomain);
      }
    }

  typedef typename subdomain_vector::const_iterator iterator;

  if(N<10) {
    int i=0;
    for(iterator iter=domains.begin(); iter != domains.end(); ++iter) {
      typedef typename std::iterator_traits<iterator>::value_type
      ::const_iterator entry_iterator;
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

  Dune::Timer watch;

  watch.reset();

  Dune::InverseOperatorResult res;

  std::cout<<"Additive Schwarz (domains vector)"<<std::endl;

  b=0;
  x=100;
  //  setBoundary(x,b,N);
#if HAVE_SUITESPARSE_UMFPACK
  std::cout << "Do testing with UMFPack" << std::endl;
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode,
      Dune::UMFPack<BCRSMat> > prec0(mat, domains, 1);
  Dune::LoopSolver<BVector> solver0(fop, prec0, 1e-2,100,2);
  solver0.apply(x,b, res);
  suite.check(res.converged) << "solver0 did not converge";

  b=0;
  x=100;
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode,
                              Dune::UMFPack<BCRSMat> >
    prec1(mat, domains, 1, false);
  Dune::LoopSolver<BVector> solver1(fop, prec1, 1e-2,100,2);
  solver1.apply(x,b, res);
  suite.check(res.converged) << "solver1 did not converge";

#endif // HAVE_SUITESPARSE_UMFPACK
#if HAVE_SUPERLU
  std::cout << "Do testing with SuperLU" << std::endl;
  x=100;
  b=0;
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode,
      Dune::SuperLU<BCRSMat> > slu_prec0(mat, domains, 1);
  Dune::LoopSolver<BVector> slu_solver(fop, slu_prec0, 1e-2,100,2);
  slu_solver.apply(x,b, res);
  suite.check(res.converged) << "slu_solver did not converge";

  x=100;
  b=0;
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode,
                              Dune::SuperLU<BCRSMat> > slu_prec1(mat, domains, 1, false);
  Dune::LoopSolver<BVector> slu_solver1(fop, slu_prec1, 1e-2,100,2);
  slu_solver1.apply(x,b, res);
  suite.check(res.converged) << "slu_solver1 did not converge";

#endif
  x=100;
  b=0;

  std::cout << "Do testing with DynamicMatrixSubdomainSolver" << std::endl;
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode,
                              Dune::DynamicMatrixSubdomainSolver<BCRSMat,BVector,BVector> > dyn_prec0(mat, domains, 1);
  Dune::LoopSolver<BVector> dyn_solver(fop, dyn_prec0, 1e-2,100,2);
  dyn_solver.apply(x,b, res);
  suite.check(res.converged) << "dyn_solver did not converge";

  std::cout<<"Additive Schwarz not on the fly (domains vector)"<<std::endl;

  b=0;
  x=100;
  //  setBoundary(x,b,N);
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::AdditiveSchwarzMode> prec0o(mat, domains, 1, false);
  Dune::LoopSolver<BVector> solver0o(fop, prec0o, 1e-2,100,2);
  solver0o.apply(x,b, res);
  suite.check(res.converged) << "solver0o did not converge";

  std::cout << "Multiplicative Schwarz (domains vector)"<<std::endl;

  b=0;
  x=100;
  //setBoundary(x,b,N);
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::MultiplicativeSchwarzMode> prec1m(mat, domains, 1);
  Dune::LoopSolver<BVector> solver1m(fop, prec1m, 1e-2,100,2);
  solver1m.apply(x,b, res);
  suite.check(res.converged) << "solver1m did not converge";

  std::cout<<"Additive Schwarz (rowToDomain vector)"<<std::endl;

  b=0;
  x=100;
  //  setBoundary(x,b,N);
  if(N<10) {
    typedef typename rowtodomain_vector::const_iterator rt_iter;
    int row=0;
    std::cout<<" row to domain"<<std::endl;
    for(rt_iter i= rowToDomain.begin(); i!= rowToDomain.end(); ++i, ++row) {
      std::cout<<"row="<<row<<": ";
      typedef typename rowtodomain_vector::value_type::const_iterator diter;
      for(diter d=i->begin(); d!=i->end(); ++d)
        std::cout<<*d<<" ";
      std::cout<<std::endl;
    }
  }

  Dune::SeqOverlappingSchwarz<BCRSMat,BVector> prec2(mat, rowToDomain, 1);
  Dune::LoopSolver<BVector> solver2(fop, prec2, 1e-2,100,2);
  solver2.apply(x,b, res);
  suite.check(res.converged) << "solver2 did not converge";

  std::cout << "Multiplicative Schwarz (rowToDomain vector)"<<std::endl;

  b=0;
  x=100;
  //setBoundary(x,b,N);
  Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::MultiplicativeSchwarzMode> prec3(mat, rowToDomain, 1);
  Dune::LoopSolver<BVector> solver3(fop, prec3, 1e-2,100,2);
  solver3.apply(x,b, res);
  suite.check(res.converged) << "solver3 did not converge";

  std::cout << "SOR"<<std::endl;

  b=0;
  x=100;
  //setBoundary(x,b,N);
  Dune::SeqSOR<BCRSMat,BVector,BVector> sor(mat, 1,1);
  Dune::LoopSolver<BVector> solver4(fop, sor, 1e-2,100,2);
  solver4.apply(x,b, res);
  suite.check(res.converged) << "solver4 did not converge";

#endif // HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK
  return suite;
}

int main(int argc, char** argv){
  Dune::TestSuite suite;
  suite.subTest(test<Dune::FieldMatrix<double, 1, 1>, Dune::FieldVector<double, 1>>(argc, argv));
  suite.subTest(test<double, double>(argc, argv));
  return suite.exit();
}
