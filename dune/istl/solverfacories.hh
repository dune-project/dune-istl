// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORIES_HH
#define DUNE_ISTL_SOLVERFACTORIES_HH

namespace Dune {
  namespace SolverFactories {
#if !__cpp_inline_variables
    namespace {
#endif

      DUNE_INLINE_VARIABLE constexpr auto cgsolver =
        [](auto lin_op, const ParameterTree& config, auto comm) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;
          using Range = typename std::decay_t<decltype(*lin_op)>::range_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = logLevelToVerbosity(log.level());

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<
            Dune::CGSolver<
              typename LinearOperator::Domain
              >
            >(*lin_op, ,*prec,reduction,max_iterations,verbose);
        };


#if !__cpp_inline_variables
    }
#endif
  }
}


#endif
