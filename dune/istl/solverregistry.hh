// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREGISTRY_HH
#define DUNE_ISTL_SOLVERREGISTRY_HH

#include <dune/istl/common/registry.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>

#define DUNE_REGISTER_DIRECT_SOLVER(name, ...)                \
  registry_put(DirectSolverTag, name, __VA_ARGS__)

#define DUNE_REGISTER_PRECONDITIONER(name, ...)                \
  registry_put(PreconditionerTag, name, __VA_ARGS__)

#define DUNE_REGISTER_ITERATIVE_SOLVER(name, ...)                \
  registry_put(IterativeSolverTag, name, __VA_ARGS__)

namespace Dune{
  /** @addtogroup ISTL_Factory
      @{
  */

  namespace {
    struct DirectSolverTag {};
    struct PreconditionerTag {};
    struct IterativeSolverTag {};
  }

  template<template<class,class,class,int>class Preconditioner, int l=1>
  auto default_preconditoner_BL_creator(){
    return [](auto tl, const auto& mat, const Dune::ParameterTree& config)
           {
             using M = typename Dune::TypeListElement<0, decltype(tl)>::type;
             using D = typename Dune::TypeListElement<1, decltype(tl)>::type;
             using R = typename Dune::TypeListElement<2, decltype(tl)>::type;
             std::shared_ptr<Dune::Preconditioner<D,R>> prec
               = std::make_shared<Preconditioner<M,D,R,l>>(mat,config);
             return prec;
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

  template<template<class...>class Solver>
  auto default_iterative_solver_creator(){
    return [](auto tl,
              const auto& op,
              const auto& sp,
              const auto& prec,
              const Dune::ParameterTree& config)
           {
             using D = typename Dune::TypeListElement<0, decltype(tl)>::type;
             using R = typename Dune::TypeListElement<1, decltype(tl)>::type;
             std::shared_ptr<Dune::InverseOperator<D,R>> solver
               = std::make_shared<Solver<D>>(op, sp, prec,config);
             return solver;
           };
  }


  /* This exception is thrown, when the requested solver is in the factory but
  cannot be instantiated for the required template parameters
  */
  class UnsupportedType : public NotImplemented {};
} // end namespace Dune

#endif // DUNE_ISTL_SOLVERREGISTRY_HH
