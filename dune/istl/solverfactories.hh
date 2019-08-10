// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORIES_HH
#define DUNE_ISTL_SOLVERFACTORIES_HH

#include <dune/istl/solvers.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/ldl.hh>
#include <dune/istl/spqr.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/cholmod.hh>

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

      template<class O>
      using _matrix_type = typename O::matrix_type;

      template<class Operator>
      auto getmat(std::shared_ptr<Operator>& op){
        typedef typename Operator::domain_type X;
        typedef typename Operator::range_type Y;
        using matrix_type = Std::detected_or_t<BCRSMatrix<FieldMatrix<double, 1, 1>>, _matrix_type, Operator>;
        std::shared_ptr<AssembledLinearOperator<matrix_type, X, Y>> assembled_op = std::dynamic_pointer_cast<AssembledLinearOperator<matrix_type, X, Y>>(op);
        if(!assembled_op)
          DUNE_THROW(Exception, "The passed solver is not of type AssembledLinearOperator");
        return assembled_op->getmat();
      }

      DUNE_INLINE_VARIABLE const auto umfpack =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          typedef std::decay_t<decltype(*lin_op)> Operator;
          typedef typename Operator::domain_type X;
          typedef typename Operator::range_type Y;
          auto mat = getmat(lin_op);
          typedef std::decay_t<decltype(mat)> MatrixType;
          int verbose = config.get("verbose", 0);
          return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<UMFPack<MatrixType>>(mat, verbose));
        };

      DUNE_INLINE_VARIABLE const auto ldl =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          typedef std::decay_t<decltype(*lin_op)> Operator;
          typedef typename Operator::domain_type X;
          typedef typename Operator::range_type Y;
          auto mat = getmat(lin_op);
          typedef std::decay_t<decltype(mat)> MatrixType;
          int verbose = config.get("verbose", 0);
          return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<LDL<MatrixType>>(mat, verbose));
        };

      DUNE_INLINE_VARIABLE const auto spqr =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          typedef std::decay_t<decltype(*lin_op)> Operator;
          typedef typename Operator::domain_type X;
          typedef typename Operator::range_type Y;
          auto mat = getmat(lin_op);
          typedef std::decay_t<decltype(mat)> MatrixType;
          int verbose = config.get("verbose", 0);
          return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<SPQR<MatrixType>>(mat, verbose));
        };

      DUNE_INLINE_VARIABLE const auto superlu =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          typedef std::decay_t<decltype(*lin_op)> Operator;
          typedef typename Operator::domain_type X;
          typedef typename Operator::range_type Y;
          auto mat = getmat(lin_op);
          typedef std::decay_t<decltype(mat)> MatrixType;
          int verbose = config.get("verbose", 0);
          bool reusevector = config.get("reusevector", true);
          return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<SuperLU<MatrixType>>(mat, verbose, reusevector));
        };

      DUNE_INLINE_VARIABLE const auto cholmod =
        [](auto lin_op, const ParameterTree& config, auto comm, auto prec) {
          typedef std::decay_t<decltype(*lin_op)> Operator;
          typedef typename Operator::domain_type X;
          typedef typename Operator::range_type Y;
          auto mat = getmat(lin_op);
          typedef std::decay_t<decltype(mat)> MatrixType;
          auto iop = std::make_shared<Cholmod<MatrixType>>();
          iop->setMatrix(mat);
          return std::dynamic_pointer_cast<InverseOperator<X,Y>>(iop);
        };
#ifndef __cpp_inline_variables
    }
#endif
  }
}


#endif
