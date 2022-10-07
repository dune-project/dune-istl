// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREGISTRY_HH
#define DUNE_ISTL_SOLVERREGISTRY_HH

#include <dune/istl/common/registry.hh>
#include <dune/istl/preconditioner.hh>
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

  /* This exception is thrown, when the requested solver is in the factory but
  cannot be instantiated for the required template parameters
  */
  class UnsupportedType : public NotImplemented {};

  class InvalidSolverFactoryConfiguration : public InvalidStateException{};
} // end namespace Dune

#endif // DUNE_ISTL_SOLVERREGISTRY_HH
