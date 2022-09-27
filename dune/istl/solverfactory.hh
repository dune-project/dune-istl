// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORY_HH
#define DUNE_ISTL_SOLVERFACTORY_HH

#include <unordered_map>
#include <functional>
#include <memory>

#include <dune/common/parametertree.hh>
#include <dune/common/singleton.hh>

#include "solverregistry.hh"
#include <dune/istl/solver.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/novlpschwarz.hh>

namespace Dune{
  /** @addtogroup ISTL_Factory
      @{
  */

  // Direct solver factory:
  template<class M, class X, class Y>
  using DirectSolverSignature = std::shared_ptr<InverseOperator<X,Y>>(const M&, const ParameterTree&);
  template<class M, class X, class Y>
  using DirectSolverFactory = Singleton<ParameterizedObjectFactory<DirectSolverSignature<M,X,Y>>>;

  // Preconditioner factory:
  template<class M, class X, class Y>
  using PreconditionerSignature = std::shared_ptr<Preconditioner<X,Y>>(const std::shared_ptr<M>&, const ParameterTree&);
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

    /** initializes the direct solvers, preconditioners and iterative solvers in
        the factories with the corresponding Matrix and Vector types.

       @tparam O the assembled linear operator type
    */
    template<class O>
    int initSolverFactories(){
      using M  = typename O::matrix_type;
      using X  = typename O::range_type;
      using Y  = typename O::domain_type;
      using TL = Dune::TypeList<M,X,Y>;
      auto& dsfac=Dune::DirectSolverFactory<M,X,Y>::instance();
      addRegistryToFactory<TL>(dsfac, DirectSolverTag{});
      auto& pfac=Dune::PreconditionerFactory<O,X,Y>::instance();
      addRegistryToFactory<TL>(pfac, PreconditionerTag{});
      using TLS = Dune::TypeList<X,Y>;
      auto& isfac=Dune::IterativeSolverFactory<X,Y>::instance();
      return addRegistryToFactory<TLS>(isfac, IterativeSolverTag{});
    }
    /** initializes the direct solvers, preconditioners and iterative solvers in
       the factories with the corresponding Matrix and Vector types.

       @tparam O the assembled linear operator type
       @tparam X the Domain type
       @tparam Y the Range type

       @deprecated Use method <code>initSolverFactories<O></code>
                   instead. This will be removed after Dune 2.8.
    */
    template<class O, class X, class Y>
    [[deprecated("Use method 'initSolverFactories<O>' instead")]]
    int initSolverFactories() {
      return initSolverFactories<O>();
    }
  } // end anonymous namespace


  template<class O, class Preconditioner>
  std::shared_ptr<Preconditioner> wrapPreconditioner4Parallel(const std::shared_ptr<Preconditioner>& prec,
                                                              const O&)
  {
    return prec;
  }

  template<class M, class X, class Y, class C, class Preconditioner>
  std::shared_ptr<Preconditioner>
  wrapPreconditioner4Parallel(const std::shared_ptr<Preconditioner>& prec,
                              const std::shared_ptr<OverlappingSchwarzOperator<M,X,Y,C> >& op)
  {
    return std::make_shared<BlockPreconditioner<X,Y,C,Preconditioner> >(prec, op->getCommunication());
  }

  template<class M, class X, class Y, class C, class Preconditioner>
  std::shared_ptr<Preconditioner>
  wrapPreconditioner4Parallel(const std::shared_ptr<Preconditioner>& prec,
                              const std::shared_ptr<NonoverlappingSchwarzOperator<M,X,Y,C> >& op)
  {
    return std::make_shared<NonoverlappingBlockPreconditioner<C,Preconditioner> >(prec, op->getCommunication());
  }

  template<class M, class X, class Y>
  std::shared_ptr<ScalarProduct<X>> createScalarProduct(const std::shared_ptr<MatrixAdapter<M,X,Y> >&)
  {
    return std::make_shared<SeqScalarProduct<X>>();
  }
  template<class M, class X, class Y, class C>
  std::shared_ptr<ScalarProduct<X>> createScalarProduct(const std::shared_ptr<OverlappingSchwarzOperator<M,X,Y,C> >& op)
  {
    return createScalarProduct<X>(op->getCommunication(), op->category());
  }

  template<class M, class X, class Y, class C>
  std::shared_ptr<ScalarProduct<X>> createScalarProduct(const std::shared_ptr<NonoverlappingSchwarzOperator<M,X,Y,C> >& op)
  {
    return createScalarProduct<X>(op->getCommunication(), op->category());
  }

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

    /** @brief get a solver from the factory
     */
    static std::shared_ptr<Solver> get(std::shared_ptr<Operator> op,
                                       const ParameterTree& config,
                                       std::shared_ptr<Preconditioner> prec = nullptr){
      std::string type = config.get<std::string>("type");
      std::shared_ptr<Solver> result;
      const matrix_type* mat = getmat(op);
      if(mat){
        if (DirectSolverFactory<matrix_type, Domain, Range>::instance().contains(type)) {
          if(op->category()!=SolverCategory::sequential){
            DUNE_THROW(NotImplemented, "The solver factory does not support parallel direct solvers!");
          }
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
        prec = PreconditionerFactory<Operator, Domain, Range>::instance().create(prec_type, op, precConfig);
        if (prec->category() != op->category() && prec->category() == SolverCategory::sequential)
          // try to wrap to a parallel preconditioner
          prec = wrapPreconditioner4Parallel(prec, op);
      }
      std::shared_ptr<ScalarProduct<Domain>> sp = createScalarProduct(op);
      result = IterativeSolverFactory<Domain, Range>::instance().create(type, op, sp, prec, config);
      return result;
    }

    /**
      @brief Construct a Preconditioner for a given Operator
     */
    static std::shared_ptr<Preconditioner> getPreconditioner(std::shared_ptr<Operator> op,
                                                             const ParameterTree& config){
      const matrix_type* mat = getmat(op);
      if(mat){
        std::string prec_type = config.get<std::string>("type");
        return PreconditionerFactory<Operator, Domain, Range>::instance().create(prec_type, op, config);
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
