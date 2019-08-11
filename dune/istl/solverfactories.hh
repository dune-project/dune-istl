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

  public:

    static auto loopsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int verbose = config.get("verbose", 0);

               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());

               return std::make_shared<Dune::LoopSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
             };
    }

    static auto gradientsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::GradientSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
             };
    }

    static auto cgsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::CGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
             };
    }

    static auto bicgstabsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::BiCGSTABSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
             };
    }

    static auto minressolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::MINRESSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose);
             };
    }

    static auto restartedgmressolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int restart = config.get("restart", 10);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::RestartedGMResSolver<X>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
             };
    }

    static auto restartedflexiblegmressolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int restart = config.get("restart", 10);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::RestartedFlexibleGMResSolver<X>>(lin_op, scalarProduct, prec, reduction, restart, max_iterations, verbose);
             };
    }

    static auto generalizedpcgsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int restart = config.get("restart", 10);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::GeneralizedPCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
             };
    }

    static auto restartedfcgsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int restart = config.get("restart", 10);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::RestartedFCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
             };
    }

    static auto completefcgsolver(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               double reduction = config.get("reduction",1e-10);
               int max_iterations = config.get("max-iterations",1000);
               int restart = config.get("restart", 10);
               int verbose = config.get("verbose", 0);
               auto scalarProduct = createScalarProduct<X>(lin_op->comm(), lin_op->category());
               return std::make_shared<Dune::CompleteFCGSolver<X>>(lin_op, scalarProduct, prec, reduction, max_iterations, verbose, restart);
             };
    }

    static auto umfpack(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               int verbose = config.get("verbose", 0);
               return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<UMFPack<matrix_type>>(getmat(lin_op), verbose));
             };
    }

    static auto ldl(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               int verbose = config.get("verbose", 0);
               return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<LDL<matrix_type>>(getmat(lin_op), verbose));
             };
    }

    static auto spqr(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               int verbose = config.get("verbose", 0);
               return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<SPQR<matrix_type>>(getmat(lin_op), verbose));
             };
    }

    static auto superlu(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               int verbose = config.get("verbose", 0);
               bool reusevector = config.get("reusevector", true);
               return std::dynamic_pointer_cast<InverseOperator<X,Y>>(std::make_shared<SuperLU<matrix_type>>(getmat(lin_op), verbose, reusevector));
             };
    }

    static auto cholmod(){
      return [](auto lin_op, const ParameterTree& config, auto prec) {
               auto iop = std::make_shared<Cholmod<matrix_type>>();
               iop->setMatrix(getmat(lin_op));
               return std::dynamic_pointer_cast<InverseOperator<X,Y>>(iop);
             };
    }
  };
}


#endif
