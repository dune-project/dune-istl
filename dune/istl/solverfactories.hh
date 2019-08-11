// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERFACTORIES_HH
#define DUNE_ISTL_SOLVERFACTORIES_HH

#include <dune/istl/solvers.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/ldl.hh>
#include <dune/istl/spqr.hh>
#include <dune/istl/superlu.hh>
#if HAVE_SUITESPARSE_CHOLMOD
#include <dune/istl/cholmod.hh>
#endif

namespace Dune {

  template<class Operator>
  class SolverFactories {
    // the following determine the matrix type or is BCRSMatrix<double,1,1> if
    // no matrix_type exists
    template<class O>
    using _matrix_type = typename O::matrix_type;
    using matrix_type = Std::detected_or_t<BCRSMatrix<FieldMatrix<double, 1, 1>>, _matrix_type, Operator>;
    typedef typename Operator::domain_type X;
    typedef typename Operator::range_type Y;
    using field_type = Simd::Scalar<typename X::field_type>;

    static auto& getmat(std::shared_ptr<Operator>& op){
      std::shared_ptr<AssembledLinearOperator<matrix_type, X, Y>> assembled_op =
        std::dynamic_pointer_cast<AssembledLinearOperator<matrix_type, X, Y>>(op);
      if(!assembled_op)
        DUNE_THROW(Exception, "The passed solver is not of type AssembledLinearOperator");
      return assembled_op->getmat();
    }

    static int getVerbose(std::shared_ptr<Operator> op,
                          const ParameterTree& config){
      auto& comm = op->comm();
      return (config.get("verboserank", 0)==comm.communicator().rank())?config.get("verbose", 1):0;
    }


  public:
    /** \brief LoopSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto loopsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<LoopSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
      };
    }

    /** \brief GradientSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - verbose (default: 1)
        - verboserank (default: 0)
     */

    static auto gradientsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<GradientSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
      };
    }

    /** \brief CGSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto cgsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<CGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
      };
    }

    /** \brief BiCGSTABSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto bicgstabsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<BiCGSTABSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
      };
    }

    /** \brief MINRESSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto minressolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<MINRESSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
      };
    }

    /** \brief RestartedGMResSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - restart (default: 10)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto restartedgmressolver(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int restart = config.get("restart", 10);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<RestartedGMResSolver<X>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
      };
    }

    /** \brief RestartedFlexibleGMResSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - restart (default: 10)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto restartedflexiblegmressolver(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int restart = config.get("restart", 10);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<RestartedFlexibleGMResSolver<X>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
      };
    }

    /** \brief GeneralizedPCGSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - restart (default: 10)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto generalizedpcgsolver(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int restart = config.get("restart", 10);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<GeneralizedPCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
      };
    }

    /** \brief RestartedFCGSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - restart (default: 10)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto restartedfcgsolver(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int restart = config.get("restart", 10);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<RestartedFCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
      };
    }

    /** \brief CompleteFCGSolver factory

        Parameters:
        - reduction (default: 1e-10)
        - max_iterations (default: 1000)
        - restart (default: 10)
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto completefcgsolver(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
        double reduction = config.get("reduction",1e-10);
        int max_iterations = config.get("max-iterations",1000);
        int restart = config.get("restart", 10);
        int verbose = SolverFactories::getVerbose(lin_op, config);
        auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
        return std::make_shared<CompleteFCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
      };
    }

    /** \brief UMFPack factory

        Parameters:
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto umfpack(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
#if HAVE_SUITESPARSE_UMFPACK
          int verbose = SolverFactories::getVerbose(lin_op, config);
          return std::make_shared<UMFPack<matrix_type>>(SolverFactories::getmat(lin_op), verbose);
#else
          DUNE_THROW(Exception, "UMFPack is not available.");
#endif
        };
    }

    /** \brief LDL factory

        Parameters:
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto ldl(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
#if HAVE_SUITESPARSE_LDL
        int verbose = SolverFactories::getVerbose(lin_op, config);
        return std::make_shared<LDL<matrix_type>>(SolverFactories::getmat(lin_op), verbose);
#else
        DUNE_THROW(Exception, "LDL is not available.");
#endif
      };
    }

    /** \brief SPQR factory

        Parameters:
        - verbose (default: 1)
        - verboserank (default: 0)
     */
    static auto spqr(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
#if HAVE_SUITESPARSE_SPQR
        int verbose = SolverFactories::getVerbose(lin_op, config);
        return std::make_shared<SPQR<matrix_type>>(SolverFactories::getmat(lin_op), verbose);
#else
        DUNE_THROW(Exception, "SPQR is not available.");
#endif
      };
    }

    /** \brief SuperLU factory

        Parameters:
        - verbose (default: 1)
        - verboserank (default: 0)
        - reusevector (default: true)
     */
    static auto superlu(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
#if HAVE_SUPERLU
        int verbose = SolverFactories::getVerbose(lin_op, config);
        bool reusevector = config.get("reusevector", true);
        return std::make_shared<SuperLU<matrix_type>>(SolverFactories::getmat(lin_op), verbose, reusevector);
#else
        DUNE_THROW(Exception, "SuperLU is not available.");
#endif
      };
    }

    /** \brief Cholmod factory

        Parameters:
     */
    static auto cholmod(){
      return [&](auto lin_op, const ParameterTree& config, auto prec)
        -> std::shared_ptr<InverseOperator<X,Y>> {
#if HAVE_SUITESPARSE_CHOLMOD
        auto iop = std::make_shared<Cholmod<matrix_type>>();
        iop->setMatrix(SolverFactories::getmat(lin_op));
        return iop;
#else
        DUNE_THROW(Exception, "Cholmod is not available.");
#endif
      };
    }
  };
}


#endif
