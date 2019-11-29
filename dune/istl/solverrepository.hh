// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREPOSITORY_HH
#define DUNE_ISTL_SOLVERREPOSITORY_HH

#include <unordered_map>
#include <functional>
#include <memory>

#include <dune/common/parametertree.hh>
#include <dune/common/singleton.hh>

#include <dune/istl/common/registry.hh>
#include <dune/istl/solver.hh>

namespace {
  struct DirectSolverTag {};
}

// typelist parameter contains matrix_type, domain_type and range_type
// parameters are matrix and ParameterTree
#define DUNE_REGISTER_DIRECT_SOLVER(name, ...)                \
  registry_put(DirectSolverTag, name, __VA_ARGS__)


template<template<class>class Solver>
auto default_direct_solver_creator(){
  return [](auto tl, const auto& mat, const Dune::ParameterTree& config)
             {
               using M = typename Dune::TypeListElement<0, decltype(tl)>::type;
               using D = typename Dune::TypeListElement<1, decltype(tl)>::type;
               using R = typename Dune::TypeListElement<2, decltype(tl)>::type;
               int verbose = config.get("verbose", 0);
               std::shared_ptr<Dune::InverseOperator<D,R>> solver
                 = std::make_shared<Solver<M>>(mat,verbose);
               return solver;
             };
}

namespace Dune{
  template<class M, class X, class Y>
  class DirectSolverFactory{
    using Signature = std::shared_ptr<InverseOperator<X,Y>>(const M&, const ParameterTree&);
    using FactoryType =  ParameterizedObjectFactory<Signature>;

  public:
    DirectSolverFactory()
    {
    }

    template<class UniqueTag=void>
    static inline void reg(){
      addRegistryToFactory<TypeList<M,X,Y>>(Singleton<FactoryType>::instance(),
                                             DirectSolverTag{});
    }

    FactoryType& instance(){
      return Singleton<FactoryType>::instance();
    }
  };

  template<class Operator>
  class SolverRepository {
    using Domain = typename Operator::domain_type;
    using Range = typename Operator::range_type;
    using Solver = Dune::InverseOperator<Domain,Range>;
    using Preconditioner = Dune::Preconditioner<Domain, Range>;

    template<class O>
    using _matrix_type = typename O::matrix_type;
    using matrix_type = Std::detected_or_t<int, _matrix_type, Operator>;
    static constexpr bool isAssembled = !std::is_same<matrix_type, int>::value;

    static const matrix_type* getmat(std::shared_ptr<Operator> op){
      std::shared_ptr<AssembledLinearOperator<matrix_type, Domain, Range>> aop
        = std::dynamic_pointer_cast<AssembledLinearOperator<matrix_type, Domain, Range>>(op);
      if(aop)
        return &aop->getmat();
      return nullptr;
    }
  public:

    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       std::shared_ptr<Preconditioner> prec = nullptr){
      std::string type = config.get<std::string>("type");
      std::shared_ptr<Solver> result;
      const matrix_type* mat = getmat(op);
      if(mat){
        try{
          result = DirectSolverFactory<matrix_type, Domain, Range>().instance().create(type, *mat, config);
        }catch(...){}
      }
      if(!result){
        DUNE_THROW(Dune::InvalidStateException, "Solver \"" << type << "\" was not found in the repository");
      }
      return result;
    }
  };

  /**
     \brief Instanciates an `InverseOperator` from an Operator and a
     configuration given in a ParameterTree.
     \param op Operator
     \param config `ParamerTree` with configuration
     \param prec Custom `Preconditioner` (optional). If not given it will be
     created with the `PreconditionerRepository` and the configuration given in
     subKey "preconditioner".

   */
  template<class Operator>
  std::shared_ptr<InverseOperator<typename Operator::domain_type,
                                  typename Operator::range_type>> getSolverFromRepository(std::shared_ptr<Operator> op,
                               const ParameterTree& config,
                               std::shared_ptr<Preconditioner<typename Operator::domain_type,
                               typename Operator::range_type>> prec = nullptr){
    return SolverRepository<Operator>::get(op, config, prec);
  }

  /**
 * @}
 */
}


#endif
