// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORIES_HH
#define DUNE_ISTL_SOLVERFACTORIES_HH

#include <dune/istl/solvers.hh>

namespace Dune {

  namespace SolverFactories {
#ifndef __cpp_inline_variables
    namespace {
#endif

      DUNE_INLINE_VARIABLE const auto loopsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::LoopSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto gradientsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::GradientSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto cgsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::CGSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto bicgstabsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::BiCGSTABSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto minressolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::MINRESSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto restartedgmressolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int restart = config.get("restart", 10);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::RestartedGMResSolver<Domain>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto restartedflexiblegmressolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int restart = config.get("restart", 10);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::RestartedFlexibleGMResSolver<Domain>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
        };

      DUNE_INLINE_VARIABLE const auto generalizedpcgsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int restart = config.get("restart", 10);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::GeneralizedPCGSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
        };

      DUNE_INLINE_VARIABLE const auto restartedfcgsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int restart = config.get("restart", 10);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::RestartedFCGSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
        };

      DUNE_INLINE_VARIABLE const auto completefcgsolver =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;

          double reduction = config.get("reduction",1e-10);
          int max_iterations = config.get("max-iterations",1000);
          int restart = config.get("restart", 10);
          int verbose = config.get("verbose", 0);

          auto scalarProduct = createScalarProduct<Domain>(comm, lin_op->category());

          return std::make_shared<Dune::CompleteFCGSolver<Domain>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
        };
#ifndef __cpp_inline_variables
    }
#endif
  }
}


#endif