// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_PRECONDITIONERFACTORIES_HH
#define DUNE_ISTL_PRECONDITIONERFACTORIES_HH

#include <dune/istl/preconditioners.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/istl/paamg/amg.hh>

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


    static Amg::Parameters getAMGParameter(const ParameterTree& config){
      using Parameters   = Dune::Amg::Parameters;
      Parameters parameters;

      if (config.hasKey("preset")){
        auto diameter = config.get("diameter",2);
        auto dim = config.get("dim",2);
        auto preset = config["preset"];
        if (preset == "isotropic")
          parameters.setDefaultValuesIsotropic(dim,diameter);
        else if (preset == "anisotropic")
          parameters.setDefaultValuesAnisotropic(dim,diameter);
        else
          DUNE_THROW(Exception,"Unknown AMG preset: " << preset);
      }

      if (config.hasKey("max-distance"))
        parameters.setMaxDistance(config.get<std::size_t>("max-distance"));

      if (config.hasKey("skip-isolated"))
        parameters.setMaxDistance(config.get<bool>("skip-isolated"));

      if (config.hasKey("min-aggregate-size"))
        parameters.setMinAggregateSize(config.get<std::size_t>("min-aggregate-size"));

      if (config.hasKey("max-aggregate-size"))
        parameters.setMinAggregateSize(config.get<std::size_t>("max-aggregate-size"));

      if (config.hasKey("max-connectivity"))
        parameters.setMinAggregateSize(config.get<std::size_t>("max-connectivity"));

      if (config.hasKey("alpha"))
        parameters.setAlpha(config.get<double>("alpha"));

      if (config.hasKey("beta"))
        parameters.setBeta(config.get<double>("beta"));

      if (config.hasKey("max-level"))
        parameters.setMaxLevel(config.get<int>("max-level"));

      if (config.hasKey("coarsen-target"))
        parameters.setCoarsenTarget(config.get<int>("coarsen-target"));

      if (config.hasKey("min-coarsen-rate"))
        parameters.setMinCoarsenRate(config.get<double>("min-coarsen-rate"));

      // TODO: accumulation mode!

      if (config.hasKey("prolongation-damping-factor"))
        parameters.setProlongationDampingFactor(config.get<double>("prolongation-damping-factor"));

      if (config.hasKey("debug-level"))
        parameters.setDebugLevel(config.get<int>("debug-level"));

      if (config.hasKey("pre-smooth-steps"))
        parameters.setNoPreSmoothSteps(config.get<std::size_t>("pre-smooth-steps"));

      if (config.hasKey("post-smooth-steps"))
        parameters.setNoPostSmoothSteps(config.get<std::size_t>("post-smooth-steps"));

      if (config.hasKey("gamma"))
        parameters.setGamma(config.get<std::size_t>("gamma"));

      if (config.hasKey("additive"))
        parameters.setAdditive(config.get<bool>("additive"));

      return parameters;
    }

    template<class O>
    struct AMGCompatibleOperator{
      using type = MatrixAdapter<matrix_type, X, Y>;
    };

    template<class GI, class LI>
    struct AMGCompatibleOperator<OverlappingSchwarzOperator<matrix_type, X, Y, OwnerOverlapCopyCommunication<GI, LI>>>{
      using type = OverlappingSchwarzOperator<matrix_type, X, Y, OwnerOverlapCopyCommunication<GI, LI>>;
    };

    template<class GI, class LI>
    struct AMGCompatibleOperator<NonoverlappingSchwarzOperator<matrix_type, X, Y, OwnerOverlapCopyCommunication<GI, LI>>>{
      using type = NonoverlappingSchwarzOperator<matrix_type, X, Y, OwnerOverlapCopyCommunication<GI, LI>>;
    };
  public:

    /** \brief Richardson factory

        SolverCategory: any

        Parameters:
        - relaxation (default: 1.0)
     */
    static auto richardson(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        return std::make_shared<Richardson<X, Y>>(relaxation);
      };
    }

    /** \brief InverseOperator2Preconditioner factory

        SolverCategory: any

        Parameters:
        - preconditioner (no default, subtree)
     */
    static auto inverseoperator2preconditioner(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        std::shared_ptr<InverseOperator<X, Y>> iop = SolverRepository<Operator>::get(lin_op, config.sub("solver"));
        return std::make_shared<InverseOperator2Preconditioner<InverseOperator<X, Y>>>(iop);
      };
    }

    /** \brief SeqSSOR factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
     */
    static auto seqssor(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        return std::make_shared<SeqSSOR<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation);
      };
    }

    /** \brief SeqSOR factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
     */
    static auto seqsor(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        return std::make_shared<SeqSOR<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation);
      };
    }

    /** \brief SeqGS factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
     */
    static auto seqgs(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        return std::make_shared<SeqGS<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation);
      };
    }

    /** \brief SeqJac factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
     */
    static auto seqjac(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        return std::make_shared<SeqJac<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation);
      };
    }

    /** \brief SeqILU factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
        - resort (default: false)
     */
    static auto seqilu(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        bool resort = config.get("resort", false);
        return std::make_shared<SeqILU<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation, resort);
      };
    }

    /** \brief SeqILDL factory

        SolverCategory: sequential

        Parameters:
        - relaxation (default: 1.0)
     */
    static auto seqildl(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        return std::make_shared<SeqILDL<matrix_type, X, Y>>(PreconditionerFactories::getmat(lin_op), relaxation);
      };
    }

    /** \brief ParSSOR factory

        SolverCategory: overlapping

        Parameters:
        - relaxation (default: 1.0)
        - iterations (default: 1)
     */
    static auto parssor(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        field_type relaxation = config.get("relaxation", 1.0);
        int iterations = config.get("iterations", 1);
        return std::make_shared<ParSSOR<matrix_type, X, Y, communication_type>>(PreconditionerFactories::getmat(lin_op), iterations, relaxation, lin_op->comm());
      };
    }

    /** \brief BlockPreconditioner factory

        SolverCategory: overlapping

        Parameters:
        - preconditioner (no default, subkey)
     */
    static auto blockpreconditioner(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        auto seq_prec = PreconditionerRepository<std::decay_t<decltype(*lin_op)>>::get(lin_op, config.sub("preconditioner"));
        return std::make_shared<BlockPreconditioner<X, Y, communication_type>>(seq_prec, lin_op->comm());
      };
    }

    /** \brief NonoverlappingBlockPreconditioner factory

        SolverCategory: nonoverlapping

        Parameters:
        - preconditioner (no default, subkey)
     */
    static auto nonoverlappingblockpreconditioner(){
      return [&](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>>{
        auto seq_prec = PreconditionerRepository<std::decay_t<decltype(*lin_op)>>::get(lin_op, config.sub("preconditioner"));
        return std::make_shared<NonoverlappingBlockPreconditioner<communication_type, Preconditioner<X,Y>>>(seq_prec, lin_op->comm());
      };
    }

    /** \brief Amg::AMG factory

        SolverCategory: any

        Parameters:
        - preset (no default, values: isotropic, anisotropic)
        - diameter (default: 2, only if preset is set)
        - dim (default: 2)
        - max-distance
        - skip-isolated
        - min-aggregate-size
        - max-aggregate-size
        - max-connectivity
        - max-connectivity
        - alpha
        - beta
        - max-level
        - coarsen-target
        - min-coarsen-rarget
        - prolongation-damping-factor
        - debug-level
        - pre-smooth-steps
        - post-smooth-steps
        - gamma (1: V-cycle, 2: W-cycle)
        - additive (true: Additive, false: Multiplicative)
        - critrion (default: symmetric, values:symmetric, unsymmetric)
        - smoother (default: ssor, values: ssor, sor, gs, jac, ilu)
        - smoother.iterations (default: 1)
        - smoother.relaxation (default: 1.0)
    */
    static auto amg(){
      auto buildForSmoother = [&](auto lin_op, const ParameterTree& config, auto criterion_type, auto smoother_type)
        -> std::shared_ptr<Preconditioner<X,Y>> {
        using Criterion = typename decltype(criterion_type)::type;
        using Smoother = typename decltype(smoother_type)::type;
        using SmootherArgs = typename Dune::Amg::SmootherTraits<Smoother>::Arguments;
        SmootherArgs smoother_args;
        smoother_args.iterations = config.get("smoother.iterations",1);
        smoother_args.relaxationFactor = config.get("smoother.relaxation",1.0);
        auto parameters = PreconditionerFactories::getAMGParameter(config);
        Criterion criterion(parameters);
        using AMGOP = typename AMGCompatibleOperator<Operator>::type;
        using AMG = Amg::AMG<AMGOP, X, Smoother, typename AMGOP::communication_type>;
        auto amgop = std::dynamic_pointer_cast<AMGOP>(lin_op);
        if(!amgop)
          DUNE_THROW(Exception, "The operator is not AMG compatible");
        return std::make_shared<AMG>(*amgop, criterion, smoother_args, amgop->comm());
      };
      auto buildForCriterion = [&, buildForSmoother](auto lin_op, const ParameterTree& config, auto criterion_type)
        -> std::shared_ptr<Preconditioner<X,Y>> {
        auto smoother = config.get("smoother", "ssor");
        if(smoother == "ssor")
          return buildForSmoother(lin_op, config, criterion_type, MetaType<Dune::SeqSSOR<matrix_type, X, Y>>{});
        else if(smoother == "sor")
          return buildForSmoother(lin_op, config, criterion_type, MetaType<Dune::SeqSOR<matrix_type, X, Y>>{});
        else if(smoother == "gs")
          return buildForSmoother(lin_op, config, criterion_type, MetaType<Dune::SeqGS<matrix_type, X, Y>>{});
        else if(smoother == "jac")
          return buildForSmoother(lin_op, config, criterion_type, MetaType<Dune::SeqJac<matrix_type, X, Y>>{});
        else if(smoother == "ilu")
          return buildForSmoother(lin_op, config, criterion_type, MetaType<Dune::SeqILU<matrix_type, X, Y>>{});
        else
          DUNE_THROW(Dune::Exception, "Unknown smoother " << smoother);
      };
      return [&, buildForCriterion](auto lin_op, const ParameterTree& config)
        -> std::shared_ptr<Preconditioner<X,Y>> {
        auto criterion = config.get("criterion","symmetric");
        if(criterion == "symmetric"){
          return buildForCriterion(lin_op, config, MetaType<Dune::Amg::SymmetricCriterion<matrix_type,Dune::Amg::FirstDiagonal>>{});
        }else if(criterion == "unsymmetric"){
          return buildForCriterion(lin_op, config, MetaType<Dune::Amg::UnSymmetricCriterion<matrix_type,Dune::Amg::FirstDiagonal>>{});
        }else{
          DUNE_THROW(Exception,"Unknown criterion: " << criterion);
        }
      };
    }
  };
}


#endif
