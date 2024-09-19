// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREGISTRY_HH
#define DUNE_ISTL_SOLVERREGISTRY_HH

#include <dune/istl/common/registry.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>

namespace Dune::Impl {
  template<class C>
  [[deprecated("DUNE_REGISTER_ITERATIVE_SOLVER is deprecated. Please use DUNE_REGISTER_SOLVER instead.")]]
  auto translateToOldIterativeSolverInterface(C oldCreator){
    return [=](auto optraits, auto&&... args){
      using OpTraits = decltype(optraits);
      using TL = TypeList<typename OpTraits::domain_type, typename OpTraits::range_type>;
      return oldCreator(TL{}, args...);
    };
  }
}

#define DUNE_REGISTER_DIRECT_SOLVER(name, ...)  \
  _Pragma ("GCC warning \"'DUNE_REGISTER_DIRECT_SOLVER' macro is deprecated and will be removed after the release of DUNE 2.9. Please use 'DUNE_REGISTER_SOLVER'\"") \
  DUNE_REGISTER_SOLVER(name, __VA_ARGS__);

#define DUNE_REGISTER_ITERATIVE_SOLVER(name, ...)  \
  _Pragma ("GCC warning \"'DUNE_REGISTER_ITERATIVE_SOLVER' macro is deprecated and will be removed after the release of DUNE 2.9. Please use 'DUNE_REGISTER_SOLVER'\"") \
  DUNE_REGISTER_SOLVER(name, Impl::translateToOldIterativeSolverInterface(__VA_ARGS__));

#define DUNE_REGISTER_PRECONDITIONER(name, ...)                \
  DUNE_REGISTRY_PUT(PreconditionerTag, name, __VA_ARGS__)

#define DUNE_REGISTER_SOLVER(name, ...)                \
  DUNE_REGISTRY_PUT(SolverTag, name, __VA_ARGS__)

namespace Dune{

  /** @addtogroup ISTL_Factory
      @{
  */

  namespace {
    struct PreconditionerTag {};
    struct SolverTag {};
  }

  //! This exception is thrown if the requested solver or preconditioner needs an assembled matrix
  class NoAssembledOperator : public InvalidStateException{};

  template<template<class,class,class,int>class Preconditioner, int blockLevel=1>
  auto defaultPreconditionerBlockLevelCreator(){
    return [](auto opInfo, const auto& linearOperator, const Dune::ParameterTree& config)
    {
      using OpInfo = std::decay_t<decltype(opInfo)>;
      using Matrix = typename OpInfo::matrix_type;
      using Domain = typename OpInfo::domain_type;
      using Range = typename OpInfo::range_type;
      std::shared_ptr<Dune::Preconditioner<Domain, Range>> preconditioner;
      if constexpr (OpInfo::isAssembled){
        const auto& A = opInfo.getAssembledOpOrThrow(linearOperator);
        // const Matrix& matrix = A->getmat();
        preconditioner
          = std::make_shared<Preconditioner<Matrix, Domain, Range, blockLevel>>(A, config);
      }else{
        DUNE_THROW(NoAssembledOperator, "Could not obtain matrix from operator. Please pass in an AssembledLinearOperator.");
      }
      return preconditioner;
    };
  }

  template<template<class,class,class>class Preconditioner>
  auto defaultPreconditionerCreator(){
    return [](auto opInfo, const auto& linearOperator, const Dune::ParameterTree& config)
    {
      using OpInfo = std::decay_t<decltype(opInfo)>;
      using Matrix = typename OpInfo::matrix_type;
      using Domain = typename OpInfo::domain_type;
      using Range = typename OpInfo::range_type;
      std::shared_ptr<Dune::Preconditioner<Domain, Range>> preconditioner;
      if constexpr (OpInfo::isAssembled){
        const auto& A = opInfo.getAssembledOpOrThrow(linearOperator);
        // const Matrix& matrix = A->getmat();
        preconditioner
          = std::make_shared<Preconditioner<Matrix, Domain, Range>>(A, config);
      }else{
        DUNE_THROW(NoAssembledOperator, "Could not obtain matrix from operator. Please pass in an AssembledLinearOperator.");
      }
      return preconditioner;
    };
  }

  template<template<class...>class Solver>
  auto defaultIterativeSolverCreator(){
    return [](auto opInfo,
              const auto& linearOperator,
              const Dune::ParameterTree& config)
    {
      using OpInfo = std::decay_t<decltype(opInfo)>;
      using Operator = typename OpInfo::operator_type;
      using Domain = typename OpInfo::domain_type;
      using Range = typename OpInfo::range_type;
      std::shared_ptr<Operator> _op = std::dynamic_pointer_cast<Operator>(linearOperator);
      std::shared_ptr<Preconditioner<Domain,Range>> preconditioner = getPreconditionerFromFactory(_op, config.sub("preconditioner"));
      std::shared_ptr<ScalarProduct<Range>> scalarProduct = opInfo.getScalarProduct(linearOperator);
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
