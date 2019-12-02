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
  struct PreconditionerTag {};
  struct IterativeSolverTag {};
}

// typelist parameter contains matrix_type, domain_type and range_type
// parameters are matrix and ParameterTree
#define DUNE_REGISTER_DIRECT_SOLVER(name, ...)                \
  registry_put(DirectSolverTag, name, __VA_ARGS__)

#define DUNE_REGISTER_PRECONDITIONER(name, ...)                \
  registry_put(PreconditionerTag, name, __VA_ARGS__)

#define DUNE_REGISTER_ITERATIVE_SOLVER(name, ...)                \
  registry_put(IterativeSolverTag, name, __VA_ARGS__)


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

template<template<class,class,class>class Preconditioner>
auto default_preconditoner_creator(){
  return [](auto tl, const auto& mat, const Dune::ParameterTree& config)
         {
           using M = typename Dune::TypeListElement<0, decltype(tl)>::type;
           using D = typename Dune::TypeListElement<1, decltype(tl)>::type;
           using R = typename Dune::TypeListElement<2, decltype(tl)>::type;
           std::shared_ptr<Dune::Preconditioner<D,R>> prec
             = std::make_shared<Preconditioner<M,D,R>>(mat,config);
           return prec;
         };
}

template<template<class>class Solver>
auto default_iterative_solver_creator(){
  return [](auto tl,
            const std::shared_ptr<Dune::LinearOperator<typename Dune::TypeListElement<0, decltype(tl)>::type, typename Dune::TypeListElement<1, decltype(tl)>::type>>& op,
            const std::shared_ptr<Dune::ScalarProduct<typename Dune::TypeListElement<0, decltype(tl)>::type>>& sp,
            const std::shared_ptr<Dune::Preconditioner<typename Dune::TypeListElement<0, decltype(tl)>::type, typename Dune::TypeListElement<1, decltype(tl)>::type>>& prec,
            const Dune::ParameterTree& config)
         {
           using D = typename Dune::TypeListElement<0, decltype(tl)>::type;
           using R = typename Dune::TypeListElement<1, decltype(tl)>::type;
           std::shared_ptr<Dune::InverseOperator<D,R>> solver
             = std::make_shared<Solver<D>>(op, sp, prec,config);
           return solver;
         };
}

namespace Dune{
  // Direct solver factory:
  template<class M, class X, class Y>
  using DirectSolverSignature = std::shared_ptr<InverseOperator<X,Y>>(const M&, const ParameterTree&);
  template<class M, class X, class Y>
  using DirectSolverFactory = Singleton<ParameterizedObjectFactory<DirectSolverSignature<M,X,Y>>>;

  template<class UniqueTag, class M, class X, class Y>
  void addRegisteredDirectSolversToFactory(UniqueTag = {}){
    using TL = Dune::TypeList<M,X,Y>;
    auto& fac=Dune::DirectSolverFactory<M,X,Y>::instance();
    addRegistryToFactory<UniqueTag, TL>(fac, DirectSolverTag{});
  }

  // Preconditioner factory:
  template<class M, class X, class Y>
  using PreconditionerSignature = std::shared_ptr<Preconditioner<X,Y>>(const M&, const ParameterTree&);
  template<class M, class X, class Y>
  using PreconditionerFactory = Singleton<ParameterizedObjectFactory<PreconditionerSignature<M,X,Y>>>;

  template<class UniqueTag, class M, class X, class Y>
  void addRegisteredPreconditionersToFactory(UniqueTag = {}){
    using TL = Dune::TypeList<M,X,Y>;
    auto& fac=Dune::PreconditionerFactory<M,X,Y>::instance();
    addRegistryToFactory<UniqueTag, TL>(fac, PreconditionerTag{});
  }

  // Iterative solver factory
  template<class X, class Y>
  using IterativeSolverSignature = std::shared_ptr<InverseOperator<X,Y>>(const std::shared_ptr<LinearOperator<X,Y>>&, const std::shared_ptr<ScalarProduct<X>>&, const std::shared_ptr<Preconditioner<X,Y>>, const ParameterTree&);
  template<class X, class Y>
  using IterativeSolverFactory = Singleton<ParameterizedObjectFactory<IterativeSolverSignature<X,Y>>>;

  template<class UniqueTag, class X, class Y>
  void addRegisteredIterativeSolversToFactory(UniqueTag = {}){
    using TL = Dune::TypeList<X,Y>;
    auto& fac=Dune::IterativeSolverFactory<X,Y>::instance();
    addRegistryToFactory<UniqueTag, TL>(fac, IterativeSolverTag{});
  }

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
          result = DirectSolverFactory<matrix_type, Domain, Range>::instance().create(type, *mat, config);
          if(result) return result;
        }catch(Dune::InvalidStateException){} // if no direct solver is found its maybe an iterative
      }
      if(!prec){
        const ParameterTree& precConfig = config.sub("preconditioner");
        std::string prec_type = precConfig.get<std::string>("type");
        try{
          prec = PreconditionerFactory<matrix_type, Domain, Range>::instance().create(prec_type, *mat, precConfig);
        }catch(Dune::InvalidStateException){
          DUNE_THROW(Dune::InvalidStateException, "Preconditioner can not be found in the factory");
        }
      }
      if(op->category()!=SolverCategory::sequential){
        DUNE_THROW(NotImplemented, "The solver repository is only implemented for sequential solvers yet!");
      }
      std::shared_ptr<ScalarProduct<Domain>> sp = std::make_shared<SeqScalarProduct<Domain>>();
      result = IterativeSolverFactory<Domain, Range>::instance().create(type, op, sp, prec, config);
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
     \param config `ParameterTree` with configuration
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
