// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef BIN_SOLVE_MM_HH
#define BIN_SOLVE_MM_HH

#include <dune/common/parametertree.hh>
#include <dune/common/timer.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/test/laplacian.hh>


Dune::ParameterTree config;

template<class OP>
void solve(const std::shared_ptr<OP>& op,
           typename OP::range_type& rhs,
           typename OP::domain_type& x,
           const Dune::ParameterTree& config,
           int verbose = 1){
  Dune::Timer t;
  if(verbose)
    std::cout << "Initializing solver... " << std::flush;
  Dune::initSolverFactories<OP>();
  auto solver = Dune::getSolverFromFactory(op, config);
  if(verbose){
    std::cout << t.elapsed() << " s" << std::endl;
    std::cout << "Solving system..." << std::flush;
  }
  t.reset();
  Dune::InverseOperatorResult res;
  solver->apply(x,rhs,res);
  if(verbose)
    std::cout << t.elapsed() << " s" << std::endl;
}

template<class Num>
std::enable_if_t<Dune::IsNumber<Num>::value> fillRandom(Num& n){
  using namespace Dune::Simd;
  for(size_t l=0;l<lanes(n);++l){
    lane(l,n) = (Scalar<Num>(std::rand()+1)/(Scalar<Num>(RAND_MAX)));
  }
}

template<class Vec>
std::enable_if_t<!Dune::IsNumber<Vec>::value> fillRandom(Vec& v){
  for(auto& n : v){
    fillRandom(n);
  }
}

#if HAVE_MPI
typedef Dune::OwnerOverlapCopyCommunication<long long> OOCOMM;
template<class Mat, class Vec, class Comm>
std::shared_ptr<OOCOMM> loadSystem(std::shared_ptr<Mat>& m,
                                   std::shared_ptr<Vec>& rhs,
                                   const Dune::ParameterTree& config,
                                   Comm comm){
  std::string matrixfilename = config.get<std::string>("matrix", "laplacian");
  std::string rhsfilename;
  if(!config.get("random_rhs", false))
    rhsfilename = config.get<std::string>("rhs");
  bool distributed = config.get("distributed", false);
  std::shared_ptr<OOCOMM> oocomm;
  if(distributed){
    oocomm = std::make_shared<OOCOMM>(MPI_COMM_WORLD);
    loadMatrixMarket(*m, matrixfilename, *oocomm);
    if(config.get("random_rhs", false)){
      rhs->resize(m->N());
      srand(42);
      fillRandom(*rhs);
    }else{
      loadMatrixMarket(*rhs, rhsfilename, *oocomm, false);
    }
  }else{
    oocomm = std::make_shared<OOCOMM>(comm);
    if(comm.rank()==0){
      if(matrixfilename != "laplacian"){
        loadMatrixMarket(*m, matrixfilename);
      }else{
        setupLaplacian(*m, config.get("N", 20));
      }
      if(config.get("random_rhs", false)){
        rhs->resize(m->N());
        fillRandom(*rhs);
      }else{
        loadMatrixMarket(*rhs, rhsfilename);
      }
    }
  }
  oocomm->remoteIndices().template rebuild<false>();
  return oocomm;
}
#else
template<class Mat, class Vec>
void loadSystem(std::shared_ptr<Mat>& m,
                std::shared_ptr<Vec>& rhs,
                const Dune::ParameterTree& config){
  std::string matrixfilename = config.get<std::string>("matrix");
  std::string rhsfilename;
  if(!config.get("random_rhs", false))
    rhsfilename = config.get<std::string>("rhs");
  loadMatrixMarket(*m, matrixfilename);
  if(config.get("random_rhs", false)){
    rhs->resize(m->N());
    fillRandom(*rhs);
  }else
    loadMatrixMarket(*rhs, rhsfilename);
}
#endif

#if HAVE_MPI
template<class Mat, class Vec>
void redistribute(std::shared_ptr<Mat>& m,
                  std::shared_ptr<Vec>& rhs,
                  std::shared_ptr<OOCOMM>& oocomm){
  typedef typename Dune::Amg::MatrixGraph<Mat> MatrixGraph;
  Dune::RedistributeInformation<OOCOMM> ri;
  std::shared_ptr<Mat> m_redist = std::make_shared<Mat>();
  std::shared_ptr<OOCOMM> oocomm_redist = std::make_shared<OOCOMM>(MPI_COMM_WORLD);
  oocomm->remoteIndices().template rebuild<false>();
  Dune::graphRepartition(MatrixGraph(*m), *oocomm,
                         oocomm->communicator().size(),
                         oocomm_redist,
                         ri.getInterface(), config.get("verbose", 1)>1);
  ri.setSetup();
  oocomm_redist->remoteIndices().template rebuild<false>();
  redistributeMatrix(*m,*m_redist, *oocomm, *oocomm_redist, ri);
  std::shared_ptr<Vec> rhs_redist = std::make_shared<Vec>(m_redist->N());
  ri.redistribute(*rhs, *rhs_redist);
  m = m_redist;
  oocomm = oocomm_redist;
  rhs = rhs_redist;
  oocomm->copyOwnerToAll(*rhs, *rhs);
}
#endif

#endif
