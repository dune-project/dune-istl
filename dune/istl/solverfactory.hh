// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORY_HH
#define DUNE_ISTL_SOLVERFACTORY_HH

#include <unordered_map>
#include <functional>
#include <memory>

#include <dune/common/parametertree.hh>
#include <dune/common/singleton.hh>

#include <dune/istl/common/registry.hh>
#include <dune/istl/solver.hh>

#define DUNE_REGISTER_DIRECT_SOLVER(name, ...)                \
  DUNE_REGISTRY_PUT(DirectSolverTag, name, __VA_ARGS__)

#define DUNE_REGISTER_PRECONDITIONER(name, ...)                \
  DUNE_REGISTRY_PUT(PreconditionerTag, name, __VA_ARGS__)

#define DUNE_REGISTER_ITERATIVE_SOLVER(name, ...)                \
  DUNE_REGISTRY_PUT(IterativeSolverTag, name, __VA_ARGS__)

namespace Dune{
  /** @addtogroup ISTL_Factory
      @{
  */

  namespace {
    struct DirectSolverTag {};
    struct PreconditionerTag {};
    struct IterativeSolverTag {};
  }

  template<template<class,class,class,int>class Preconditioner, int blockLevel=1>
  auto defaultPreconditionerBlockLevelCreator(){
    return [](auto typeList, const auto& matrix, const Dune::ParameterTree& config)
           {
             using Matrix = typename Dune::TypeListElement<0, decltype(typeList)>::type;
             using Domain = typename Dune::TypeListElement<1, decltype(typeList)>::type;
             using Range = typename Dune::TypeListElement<2, decltype(typeList)>::type;
             std::shared_ptr<Dune::Preconditioner<Domain, Range>> preconditioner
               = std::make_shared<Preconditioner<Matrix, Domain, Range, blockLevel>>(matrix, config);
             return preconditioner;
           };
  }

  template<template<class,class,class>class Preconditioner>
  auto defaultPreconditionerCreator(){
    return [](auto typeList, const auto& matrix, const Dune::ParameterTree& config)
           {
             using Matrix = typename Dune::TypeListElement<0, decltype(typeList)>::type;
             using Domain = typename Dune::TypeListElement<1, decltype(typeList)>::type;
             using Range = typename Dune::TypeListElement<2, decltype(typeList)>::type;
             std::shared_ptr<Dune::Preconditioner<Domain, Range>> preconditioner
               = std::make_shared<Preconditioner<Matrix, Domain, Range>>(matrix, config);
             return preconditioner;
           };
  }

  template<template<class...>class Solver>
  auto defaultIterativeSolverCreator(){
    return [](auto typeList,
              const auto& linearOperator,
              const auto& scalarProduct,
              const auto& preconditioner,
              const Dune::ParameterTree& config)
           {
             using Domain = typename Dune::TypeListElement<0, decltype(typeList)>::type;
             using Range = typename Dune::TypeListElement<1, decltype(typeList)>::type;
             std::shared_ptr<Dune::InverseOperator<Domain, Range>> solver
               = std::make_shared<Solver<Domain>>(linearOperator, scalarProduct, preconditioner, config);
             return solver;
           };
  }

  // Direct solver factory:
  template<class M, class X, class Y>
  using DirectSolverSignature = std::shared_ptr<InverseOperator<X,Y>>(const M&, const ParameterTree&);
  template<class M, class X, class Y>
  using DirectSolverFactory = Singleton<ParameterizedObjectFactory<DirectSolverSignature<M,X,Y>>>;

  // Preconditioner factory:
  template<class M, class X, class Y>
  using PreconditionerSignature = std::shared_ptr<Preconditioner<X,Y>>(const M&, const ParameterTree&);
  template<class M, class X, class Y>
  using PreconditionerFactory = Singleton<ParameterizedObjectFactory<PreconditionerSignature<M,X,Y>>>;

  // Iterative solver factory
  template<class X, class Y>
  using IterativeSolverSignature = std::shared_ptr<InverseOperator<X,Y>>(const std::shared_ptr<LinearOperator<X,Y>>&, const std::shared_ptr<ScalarProduct<X>>&, const std::shared_ptr<Preconditioner<X,Y>>, const ParameterTree&);
  template<class X, class Y>
  using IterativeSolverFactory = Singleton<ParameterizedObjectFactory<IterativeSolverSignature<X,Y>>>;

  // initSolverFactories differs in different compilation units, so we have it
  // in an anonymous namespace
  namespace {

    /* initializes the direct solvers, preconditioners and iterative solvers in
       the factories with the corresponding Matrix and Vector types.

       @tparam M the Matrix type
       @tparam X the Domain type
       @tparam Y the Range type
    */
    template<class M, class X, class Y>
    int initSolverFactories(){
      using TL = Dune::TypeList<M,X,Y>;
      auto& dsfac=Dune::DirectSolverFactory<M,X,Y>::instance();
      addRegistryToFactory<TL>(dsfac, DirectSolverTag{});
      auto& pfac=Dune::PreconditionerFactory<M,X,Y>::instance();
      addRegistryToFactory<TL>(pfac, PreconditionerTag{});
      using TLS = Dune::TypeList<X,Y>;
      auto& isfac=Dune::IterativeSolverFactory<X,Y>::instance();
      return addRegistryToFactory<TLS>(isfac, IterativeSolverTag{});
    }
  } // end anonymous namespace

  /* This exception is thrown, when the requested solver is in the factory but
  cannot be instantiated for the required template parameters
  */
  class UnsupportedType : public NotImplemented {};

  /**
     @brief Factory to assembly solvers configured by a `ParameterTree`.

     Example ini File that can be passed in to construct a CGSolver with a SSOR
     preconditioner:
     \verbatim
     type = cgsolver
     verbose = 1
     maxit = 1000
     reduction = 1e-5

     [preconditioner]
     type = ssor
     iterations = 1
     relaxation = 1
     \endverbatim

     \tparam Operator type of the operator, necessary to deduce the matrix type etc.
   */
  template<class Operator>
  class SolverFactory {
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

    /* @brief get a solver from the factory
     */
    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       std::shared_ptr<Preconditioner> prec = nullptr){
      std::string type = config.get<std::string>("type");
      std::shared_ptr<Solver> result;
      const matrix_type* mat = getmat(op);
      if(mat){
        if (DirectSolverFactory<matrix_type, Domain, Range>::instance().contains(type)) {
          result = DirectSolverFactory<matrix_type, Domain, Range>::instance().create(type, *mat, config);
          return result;
        }
      }
      // if no direct solver is found it might be an iterative solver
      if (!IterativeSolverFactory<Domain, Range>::instance().contains(type)) {
        DUNE_THROW(Dune::InvalidStateException, "Solver not found in the factory.");
      }
      if(!prec){
        const ParameterTree& precConfig = config.sub("preconditioner");
        std::string prec_type = precConfig.get<std::string>("type");
        prec = PreconditionerFactory<matrix_type, Domain, Range>::instance().create(prec_type, *mat, precConfig);
      }
      if(op->category()!=SolverCategory::sequential){
        DUNE_THROW(NotImplemented, "The solver factory is currently only implemented for sequential solvers!");
      }
      std::shared_ptr<ScalarProduct<Domain>> sp = std::make_shared<SeqScalarProduct<Domain>>();
      result = IterativeSolverFactory<Domain, Range>::instance().create(type, op, sp, prec, config);
      return result;
    }

    /*
      @brief Construct a Preconditioner for a given Operator
     */
    static std::shared_ptr<Preconditioner> getPreconditioner(std::shared_ptr<Operator> op,
                                                             const ParameterTree& config){
      const matrix_type* mat = getmat(op);
      if(mat){
        std::string prec_type = config.get<std::string>("type");
        return PreconditionerFactory<matrix_type, Domain, Range>::instance().create(prec_type, *mat, config);
      }else{
        DUNE_THROW(InvalidStateException, "Could not obtain matrix from operator. Please pass in an AssembledLinearOperator.");
      }
    }
  };

  /**
     \brief Instantiates an `InverseOperator` from an Operator and a
     configuration given as a ParameterTree.
     \param op Operator
     \param config `ParameterTree` with configuration
     \param prec Custom `Preconditioner` (optional). If not given it will be
     created with the `PreconditionerFactory` and the configuration given in
     subKey "preconditioner".

   */
  template<class Operator>
  std::shared_ptr<InverseOperator<typename Operator::domain_type,
                                  typename Operator::range_type>> getSolverFromFactory(std::shared_ptr<Operator> op,
                               const ParameterTree& config,
                               std::shared_ptr<Preconditioner<typename Operator::domain_type,
                               typename Operator::range_type>> prec = nullptr){
    return SolverFactory<Operator>::get(op, config, prec);
  }

  /**
 * @}
 */
} // end namespace Dune


#endif
