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
    template<typename T>
    using _communication_type = typename T::communication_type;
    using Communication = Std::detected_or_t<Amg::SequentialInformation, _communication_type, Operator>;
    using FactoryType = std::function<std::shared_ptr<Preconditioner>(std::shared_ptr<Operator>,
                                                                      const ParameterTree&,
                                                                      const Communication&)>;
  public:
    using RepositoryType = std::unordered_map<std::string, FactoryType>;

    static RepositoryType repositoryInstance(){
      static RepositoryType repository = {
                                          {"richardson", PreconditionerFactories::richardson}
      };
      return repository;
    }

    static void add(const std::string& name, FactoryType factory){
      repositoryInstance().emplace(name, std::move(factory));
    }

    static std::shared_ptr<Preconditioner> get(std::shared_ptr<Operator> op,
                                               const ParameterTree& config,
                                               const Communication& comm){
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
      return fac(op, config, comm);
    }
  };

  template<class Operator>
  class SolverRepository {
    using Domain = typename Operator::domain_type;
    using Range = typename Operator::range_type;
    using Solver = Dune::InverseOperator<Domain,Range>;
    using Preconditioner = Dune::Preconditioner<Domain, Range>;

    template<typename T>
    using _communication_type = typename T::communication_type;
    using Communication = Std::detected_or_t<Amg::SequentialInformation, _communication_type, Operator>;
    using FactoryType = std::function<std::shared_ptr<Solver>(std::shared_ptr<Operator>,
                                                              const ParameterTree&,
                                                              const Communication&,
                                                              std::shared_ptr<Preconditioner>)>;
  public:
    using RepositoryType = std::unordered_map<std::string, FactoryType>;

    static RepositoryType repositoryInstance(){
      static RepositoryType repository = {
                                          {"loopsolver", SolverFactories::loopsolver},
                                          {"gradientsolver", SolverFactories::gradientsolver},
                                          {"cgsolver", SolverFactories::cgsolver},
                                          {"bicgstabsolver", SolverFactories::bicgstabsolver},
                                          {"minressolver", SolverFactories::minressolver},
                                          {"restartedgmressolver", SolverFactories::restartedgmressolver},
                                          {"restartedflexiblegmressolver", SolverFactories::restartedflexiblegmressolver},
                                          {"generalizedpcgsolver", SolverFactories::generalizedpcgsolver},
                                          {"restartedfcgsolver", SolverFactories::restartedfcgsolver},
                                          {"completefcgsolver", SolverFactories::completefcgsolver},
                                          {"umfpack", SolverFactories::umfpack},
                                          {"ldl", SolverFactories::ldl},
                                          {"spqr", SolverFactories::spqr},
                                          {"superlu", SolverFactories::superlu},
                                          {"cholmod", SolverFactories::cholmod}
      };
      return repository;
    }

    static void add(const std::string& name, FactoryType factory){
      repositoryInstance().emplace(name, std::move(factory));
    }

    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       std::shared_ptr<Preconditioner> prec = nullptr,
                                       const Communication& comm = {}){
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
        prec = PreconditionerRepository<Operator>::get(op, config.sub("preconditioner"), comm);
      }
      return fac(op, config, comm, prec);
    }
  };
}


#endif
