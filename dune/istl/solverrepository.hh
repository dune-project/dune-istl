// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREPOSITORY_HH
#define DUNE_ISTL_SOLVERREPOSITORY_HH

#include <unordered_map>
#include <functional>

#include <dune/istl/preconditionerfactories.hh>
#include <dune/istl/solverfactories.hh>
#include <dune/istl/paamg/pinfo.hh>

namespace Dune{

  template<class Operator>
  class PreconditionerRepository {
    using Domain = typename Operator::domain_type;
    using Range = typename Operator::range_type;
    using Preconditioner = Dune::Preconditioner<Domain, Range>;
    using FactoryType = std::function<std::shared_ptr<Preconditioner>(std::shared_ptr<Operator>,
                                                                      const ParameterTree&)>;
  public:
    using RepositoryType = std::unordered_map<std::string, FactoryType>;

    static RepositoryType repositoryInstance(){
      using Factories = PreconditionerFactories<Operator>;
      static RepositoryType repository = {
                                          {"richardson", Factories::richardson()},
                                          {"inverseoperator2preconditioner", Factories::inverseoperator2preconditioner()},
                                          {"seqssor", Factories::seqssor()},
                                          {"seqsor", Factories::seqsor()},
                                          {"seqgs", Factories::seqgs()},
                                          {"seqjac", Factories::seqjac()},
                                          {"seqilu", Factories::seqilu()},
                                          {"seqildl", Factories::seqildl()},
                                          {"parssor", Factories::parssor()},
                                          {"blockpreconditioner", Factories::blockpreconditioner()}
      };
      return repository;
    }

    static void add(const std::string& name, FactoryType factory){
      repositoryInstance().emplace(name, std::move(factory));
    }

    static std::shared_ptr<Preconditioner> get(std::shared_ptr<Operator> op,
                                               const ParameterTree& config){
      std::string type;
      try{
        type = config.template get<std::string>("type");
      }catch(RangeError&){
        DUNE_THROW(Exception, "SolverRepository: \"type\" is not set in the config");
      }
      FactoryType fac;
      try{
        fac = repositoryInstance().at(type);
      }catch(std::out_of_range&){
        DUNE_THROW(Exception, "Could not find preconditioner \"" << type <<  "\" in PreconditionerRepository");
      }
      return fac(op, config);
    }
  };

  template<class Operator>
  class SolverRepository {
    using Domain = typename Operator::domain_type;
    using Range = typename Operator::range_type;
    using Solver = Dune::InverseOperator<Domain,Range>;
    using Preconditioner = Dune::Preconditioner<Domain, Range>;

    using FactoryType = std::function<std::shared_ptr<Solver>(std::shared_ptr<Operator>,
                                                              const ParameterTree&,
                                                              std::shared_ptr<Preconditioner>)>;
  public:
    using RepositoryType = std::unordered_map<std::string, FactoryType>;

    static RepositoryType repositoryInstance(){
      using Factories = SolverFactories<Operator>;
      static RepositoryType repository = {
                                          {"loopsolver", Factories::loopsolver()},
                                          {"gradientsolver", Factories::gradientsolver()},
                                          {"cgsolver", Factories::cgsolver()},
                                          {"bicgstabsolver", Factories::bicgstabsolver()},
                                          {"minressolver", Factories::minressolver()},
                                          {"restartedgmressolver", Factories::restartedgmressolver()},
                                          {"restartedflexiblegmressolver", Factories::restartedflexiblegmressolver()},
                                          {"generalizedpcgsolver", Factories::generalizedpcgsolver()},
                                          {"restartedfcgsolver", Factories::restartedfcgsolver()},
                                          {"completefcgsolver", Factories::completefcgsolver()},
                                          {"umfpack", Factories::umfpack()},
                                          {"ldl", Factories::ldl()},
                                          {"spqr", Factories::spqr()},
                                          {"superlu", Factories::superlu()},
                                          {"cholmod", Factories::cholmod()}
      };
      return repository;
    }

    static void add(const std::string& name, FactoryType factory){
      repositoryInstance().emplace(name, std::move(factory));
    }

    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       std::shared_ptr<Preconditioner> prec = nullptr){
      std::string type;
      try{
        type = config.template get<std::string>("type");
      }catch(RangeError&){
        DUNE_THROW(Exception, "SolverRepository: \"type\" is not set in the config");
      }
      FactoryType fac;
      try{
        fac = repositoryInstance().at(type);
      }catch(std::out_of_range&){
        DUNE_THROW(Exception, "Could not find solver \"" << type <<  "\" in SolverRepository");
      }

      if (!prec && config.hasSub("preconditioner")){
        prec = PreconditionerRepository<Operator>::get(op, config.sub("preconditioner"));
      }
      return fac(op, config, prec);
    }
  };

  template<class Operator>
  auto getSolverFromRepository(std::shared_ptr<Operator> op,
                               const ParameterTree& config,
                               std::shared_ptr<Preconditioner<typename Operator::domain_type,
                               typename Operator::range_type>> prec = nullptr){
    return SolverRepository<Operator>::get(op, config, prec);
  }
}


#endif
