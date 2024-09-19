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
#include <dune/common/std/type_traits.hh>
#include <dune/common/singleton.hh>
#include <dune/common/parameterizedobject.hh>

#include <dune/istl/solverregistry.hh>
#include <dune/istl/common/registry.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/novlpschwarz.hh>

namespace Dune{
  /** @addtogroup ISTL_Factory
      @{
  */

  // Preconditioner factory:
  template<class OP>
  using PreconditionerSignature = std::shared_ptr<Preconditioner<typename OP::domain_type,typename OP::range_type>>(const std::shared_ptr<OP>&, const ParameterTree&);
  template<class OP>
  using PreconditionerFactory = Singleton<ParameterizedObjectFactory<PreconditionerSignature<OP>>>;

  // Iterative solver factory
  template<class OP>
  using SolverSignature = std::shared_ptr<InverseOperator<typename OP::domain_type,typename OP::range_type>>(const std::shared_ptr<OP>&, const ParameterTree&);
  template<class OP>
  using SolverFactory = Singleton<ParameterizedObjectFactory<SolverSignature<OP>>>;

  template<class Operator>
  struct OperatorTraits{
  private:
    template<class O>
    using _matrix_type = typename O::matrix_type;
    template<class O>
    using _comm_type = typename O::communication_type;
  public:
    using domain_type = typename Operator::domain_type;
    using range_type = typename Operator::range_type;
    using operator_type = Operator;
    using solver_type = InverseOperator<domain_type, range_type>;
    using matrix_type = Std::detected_or_t<int, _matrix_type, Operator>;
    static constexpr bool isAssembled = !std::is_same<matrix_type, int>::value;
    using comm_type = Std::detected_or_t<int, _comm_type, Operator>;
    static constexpr bool isParallel = !std::is_same<comm_type, int>::value;

    static const std::shared_ptr<AssembledLinearOperator<matrix_type, domain_type, range_type>>
    getAssembledOpOrThrow(std::shared_ptr<LinearOperator<domain_type, range_type>> op){
      std::shared_ptr<AssembledLinearOperator<matrix_type, domain_type, range_type>> aop
        = std::dynamic_pointer_cast<AssembledLinearOperator<matrix_type, domain_type, range_type>>(op);
      if(aop)
          return aop;
      DUNE_THROW(NoAssembledOperator, "Failed to cast to AssembledLinearOperator. Please pass in an AssembledLinearOperator.");
    }

    static const comm_type& getCommOrThrow(std::shared_ptr<LinearOperator<domain_type, range_type>> op){
      std::shared_ptr<Operator> _op
        = std::dynamic_pointer_cast<Operator>(op);
      if constexpr (isParallel){
        return _op->getCommunication();
      }else{
        DUNE_THROW(NoAssembledOperator, "Could not obtain communication object from operator. Please pass in a parallel operator.");
      }
    }

    static std::shared_ptr<ScalarProduct<domain_type>> getScalarProduct(std::shared_ptr<LinearOperator<domain_type, range_type>> op)
    {
      if constexpr (isParallel){
        return createScalarProduct<domain_type>(getCommOrThrow(op), op->category());
      }else{
        return std::make_shared<SeqScalarProduct<domain_type>>();
      }
    }
  };

  template<class Operator>
  struct [[deprecated("Please change to the new solverfactory interface.")]]
  TypeListElement<0, OperatorTraits<Operator>>{
    using type [[deprecated]] = typename OperatorTraits<Operator>::matrix_type;
    using Type [[deprecated]] = type;
  };
  template<class Operator>
  struct [[deprecated("Please change to the new solverfactory interface.")]]
  TypeListElement<1, OperatorTraits<Operator>>{
    using type [[deprecated]] = typename OperatorTraits<Operator>::domain_type;
    using Type [[deprecated]] = type;
  };
  template<class Operator>
  struct [[deprecated("Please change to the new solverfactory interface.")]]
  TypeListElement<2, OperatorTraits<Operator>>{
    using type [[deprecated]] = typename OperatorTraits<Operator>::range_type;
    using Type [[deprecated]] = type;
  };

  // initSolverFactories differs in different compilation units, so we have it
  // in an anonymous namespace
  namespace {

    /** initializes the preconditioners and solvers in
        the factories with the corresponding Operator and Vector types.

       @tparam O the assembled linear operator type
    */
    template<class O>
    int initSolverFactories(){
      using OpInfo = OperatorTraits<O>;
      auto& pfac=Dune::PreconditionerFactory<O>::instance();
      addRegistryToFactory<OpInfo>(pfac, PreconditionerTag{});
      auto& isfac=Dune::SolverFactory<O>::instance();
      return addRegistryToFactory<OpInfo>(isfac, SolverTag{});
    }
  } // end anonymous namespace

  /**
     \brief Instantiates an `InverseOperator` from an Operator and a
     configuration given as a ParameterTree.
     \param op Operator
     \param config `ParameterTree` with configuration

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
  std::shared_ptr<InverseOperator<typename Operator::domain_type,
                                  typename Operator::range_type>>
  getSolverFromFactory(std::shared_ptr<Operator> op, const ParameterTree& config,
                       std::shared_ptr<Preconditioner<typename Operator::domain_type, typename Operator::range_type>> prec = nullptr)
  {
    if(prec){
      PreconditionerFactory<Operator>::instance().define("__passed at runtime__",
                                                         [=](auto...){
                                                           return prec;
                                                         });
      ParameterTree config_tmp = config;
      config_tmp.sub("preconditioner")["type"] = std::string("__passed at runtime__");
      return SolverFactory<Operator>::instance().
        create(config.get<std::string>("type"),op, config_tmp);
    }
    return SolverFactory<Operator>::instance().
      create(config.get<std::string>("type"),op, config);
  }

  class UnknownSolverCategory : public InvalidStateException{};
  /**
     @brief Construct a Preconditioner for a given Operator
  */
  template<class Operator>
  std::shared_ptr<Preconditioner<typename Operator::domain_type,
                                 typename Operator::range_type>>
  getPreconditionerFromFactory(std::shared_ptr<Operator> op,
                               const ParameterTree& config){
    using Domain = typename Operator::domain_type;
    using Range = typename Operator::range_type;
    std::string prec_type = config.get<std::string>("type");
    std::shared_ptr<Preconditioner<typename Operator::domain_type,
                                 typename Operator::range_type>> prec = PreconditionerFactory<Operator>::instance().create(prec_type, op, config);
    if constexpr (OperatorTraits<Operator>::isParallel){
      using Comm = typename OperatorTraits<Operator>::comm_type;
      const Comm& comm = OperatorTraits<Operator>::getCommOrThrow(op);
      if(op->category() == SolverCategory::overlapping && prec->category() == SolverCategory::sequential)
        return std::make_shared<BlockPreconditioner<Domain,Range,Comm> >(prec, comm);
      else if(op->category() == SolverCategory::nonoverlapping && prec->category() == SolverCategory::sequential)
        return std::make_shared<NonoverlappingBlockPreconditioner<Comm, Preconditioner<Domain, Range>> >(prec, comm);
    }
    return prec;
  }

  /**
 * @}
 */
} // end namespace Dune


#endif
