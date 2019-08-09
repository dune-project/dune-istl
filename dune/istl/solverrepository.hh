// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREPOSITORY_HH
#define DUNE_ISTL_SOLVERREPOSITORY_HH

namespace Dune{

  template<class Operator>
  class Scala

  template<class Operator>
  class SolverRepositiory {
    using Solver = Dune::InverseOperator<Domain,Range>;
    using Communication = Operator::communication_type;
    using FactoryType = std::function<std::shared_ptr<Solver>(std::shared_ptr<Operator>,
                                             const ParameterTree&,
                                             Communication&)>;
  public:
    using RepositoryType = std::unordered_map<std::string, FactoryType>;

    static RepositoryType repositoryInstance(){
      static RepositoryType repository = {
                                          {"cg": cgsolver}
      };
      return repository;
    }

    static void add(const std::string& name, FactoryType factory){
      repositoryInstance().emplace(name, std::move(factory));
    }

    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       Communication& comm){
      std::string type;
      try{
        type = config.template get<std::string>("type");
      }catch(RangeError&){
        DUNE_THROW(Exception, "SolverRepository: \"type\" is not set in the config");
      }
      Factory fac;
      try{
        fac = repositoryInstance().at(type);
      }catch(std::out_of_range&){
        DUNE_THROW(Exception, "Could not find one step method \"" << name <<  "\"");
      }
      return fac(op, config, comm);
    }
  };
}


#endif
