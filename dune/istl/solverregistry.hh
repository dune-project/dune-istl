// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERREGISTRY_HH
#define DUNE_ISTL_SOLVERREGISTRY_HH

#include <dune/common/classname.hh>
#include <dune/istl/common/registry.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>

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

  /* This exception is thrown, when the requested solver is in the factory but
  cannot be instantiated for the required template parameters
  */
  class UnsupportedType : public NotImplemented {};

  namespace Impl {
    /* The sequential relaxation preconditioners registered via
       defaultPreconditionerBlockLevelCreator (ssor, sor, gs, jac, dilu, ilu)
       and defaultPreconditionerCreator (ildl) operate by iterating over the
       rows and columns of the matrix. Registered creators are instantiated
       for every operator type passed to initSolverFactories(), whether or not
       the preconditioner is ever selected, so they must not hard-error for
       matrix types that do not provide this interface (e.g. GPU-resident
       matrices). This concept detects the required interface; creators use it
       to throw UnsupportedType at runtime instead.
     */
    template<class M>
    concept RowIterableMatrix = requires(const M& m)
      {
        typename M::ConstRowIterator;
        typename M::ConstColIterator;
        { m.begin().index() };            // row iterators provide their row index
        { (*m.begin()).begin().index() }; // column iterators provide their column index
      };
  } // namespace Impl

  template<template<class,class,class,int>class Preconditioner, int blockLevel=1>
  auto defaultPreconditionerBlockLevelCreator(){
    return [](auto opInfo, const auto& linearOperator, const Dune::ParameterTree& config)
    {
      using OpInfo = std::decay_t<decltype(opInfo)>;
      using Matrix = typename OpInfo::matrix_type;
      using Domain = typename OpInfo::domain_type;
      using Range = typename OpInfo::range_type;
      std::shared_ptr<Dune::Preconditioner<Domain, Range>> preconditioner;
      if constexpr (!OpInfo::isAssembled){
        DUNE_THROW(NoAssembledOperator, "Could not obtain matrix from operator. Please pass in an AssembledLinearOperator.");
      } else if constexpr (!Impl::RowIterableMatrix<Matrix>) {
        DUNE_THROW(UnsupportedType,
                   "This preconditioner iterates over the matrix rows and columns, "
                   "which is not supported by " << className<Matrix>() << ".");
      } else {
        const auto& A = opInfo.getAssembledOpOrThrow(linearOperator);
        // const Matrix& matrix = A->getmat();
        preconditioner
          = std::make_shared<Preconditioner<Matrix, Domain, Range, blockLevel>>(A, config);
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
      if constexpr (!OpInfo::isAssembled){
        DUNE_THROW(NoAssembledOperator, "Could not obtain matrix from operator. Please pass in an AssembledLinearOperator.");
      } else if constexpr (!Impl::RowIterableMatrix<Matrix>) {
        DUNE_THROW(UnsupportedType,
                   "This preconditioner iterates over the matrix rows and columns, "
                   "which is not supported by " << className<Matrix>() << ".");
      } else {
        const auto& A = opInfo.getAssembledOpOrThrow(linearOperator);
        // const Matrix& matrix = A->getmat();
        preconditioner
          = std::make_shared<Preconditioner<Matrix, Domain, Range>>(A, config);
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

  class InvalidSolverFactoryConfiguration : public InvalidStateException{};
} // end namespace Dune

#endif // DUNE_ISTL_SOLVERREGISTRY_HH
