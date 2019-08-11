// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_PRECONDITIONERFACTORIES_HH
#define DUNE_ISTL_PRECONDITIONERFACTORIES_HH

#include <dune/istl/preconditioners.hh>
#include <dune/istl/schwarz.hh>

namespace Dune {

  template<class Operator>
  class SolverRepository;

  template<class>
  class PreconditionerRepository;


  template<class Operator>
  class PreconditionerFactories {
    // the following determine the matrix type or is BCRSMatrix<double,1,1> if
    // no matrix_type exists
    template<class O>
    using _matrix_type = typename O::matrix_type;
    using matrix_type = Std::detected_or_t<BCRSMatrix<FieldMatrix<double, 1, 1>>, _matrix_type, Operator>;
    typedef typename Operator::domain_type X;
    typedef typename Operator::range_type Y;
    using field_type = Simd::Scalar<typename X::field_type>;
    using communication_type = typename Operator::communication_type;

    static auto& getmat(std::shared_ptr<Operator>& op){
      std::shared_ptr<AssembledLinearOperator<matrix_type, X, Y>> assembled_op =
        std::dynamic_pointer_cast<AssembledLinearOperator<matrix_type, X, Y>>(op);
      if(!assembled_op)
        DUNE_THROW(Exception, "The passed solver is not of type AssembledLinearOperator");
      return assembled_op->getmat();
    }

  public:
    static auto richardson(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               return std::make_shared<Dune::Richardson<X, Y>>(relaxation);
             };
    }

    static auto inverseoperator2preconditioner(){
      return [](auto lin_op, const ParameterTree& config) {
               std::shared_ptr<InverseOperator<X, Y>> iop = SolverRepository<Operator>::get(lin_op, config.sub("solver"));
               return std::make_shared<Dune::InverseOperator2Preconditioner<InverseOperator<X, Y>>>(iop);
             };
    }

    static auto seqssor(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               return std::make_shared<Dune::SeqSSOR<matrix_type, X, Y>>(getmat(lin_op), iterations, relaxation);
             };
    }

    static auto seqsor(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               return std::make_shared<Dune::SeqSOR<matrix_type, X, Y>>(getmat(lin_op), iterations, relaxation);
             };
    }

    static auto seqgs(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               return std::make_shared<Dune::SeqGS<matrix_type, X, Y>>(getmat(lin_op), iterations, relaxation);
             };
    }

    static auto seqjac(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               return std::make_shared<Dune::SeqJac<matrix_type, X, Y>>(getmat(lin_op), iterations, relaxation);
             };
    }

    static auto seqilu(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               bool resort = config.get("resort", false);
               return std::make_shared<Dune::SeqILU<matrix_type, X, Y>>(getmat(lin_op), iterations, relaxation, resort);
             };
    }

    static auto seqildl(){
      return [](auto lin_op, const ParameterTree& config) {
               field_type relaxation = config.get("relaxation", 1.0);
               return std::make_shared<Dune::SeqILDL<matrix_type, X, Y>>(getmat(lin_op), relaxation);
             };
    }

    static auto parssor(){
      return [](auto lin_op, const ParameterTree& config){
               field_type relaxation = config.get("relaxation", 1.0);
               int iterations = config.get("iterations", 1);
               return std::make_shared<Dune::ParSSOR<matrix_type, X, Y, communication_type>>(getmat(lin_op), iterations, relaxation, lin_op->comm());
             };
    }

    static auto blockpreconditioner(){
      return [](auto lin_op, const ParameterTree& config){
               auto seq_prec = PreconditionerRepository<std::decay_t<decltype(*lin_op)>>::get(lin_op, config.sub("preconditioner"));
               return std::make_shared<Dune::BlockPreconditioner<X, Y, communication_type>>(seq_prec, lin_op->comm());
             };
    }
  };
}


#endif
