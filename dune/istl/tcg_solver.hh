#ifndef DUNE_TCG_SOLVER_HH
#define DUNE_TCG_SOLVER_HH

#include <iostream>
#include <utility>

#include <dune/common/typetraits.hh>
#include "dune/istl/cg_solver.hh"
#include "dune/istl/generic_iterative_method.hh"
#include "dune/istl/generic_step.hh"
#include "dune/istl/operator_type.hh"
#include "dune/istl/relative_energy_termination_criterion.hh"
#include "dune/istl/mixins/verbosity.hh"

namespace Dune
{
  /// @cond
  namespace TCGSpec
  {
    /// Cache object for the truncated conjugate gradient method.
    template <class Domain, class Range>
    struct Cache : CGSpec::Cache<Domain,Range>
    {
      template <class... Args>
      Cache(Args&&... args)
        : CGSpec::Cache<Domain,Range>( std::forward<Args>(args)... )
      {}

      void reset(LinearOperator<Domain,Range>* A,
                Preconditioner<Domain,Range>* P,
                ScalarProduct<Domain>* sp)
      {
        CGSpec::Cache<Domain,Range>::reset(A,P,sp);
        doTerminate = false;
      }

      OperatorType operatorType = OperatorType::POSITIVE_DEFINITE;
      bool doTerminate = false;
      bool performBlindUpdate = true;
    };

    /// "Truncated Conjugate Gradients"
    class Name
    {
    public:
      std::string name() const
      {
        return "Truncated Conjugate Gradients";
      }
    };


    /// Extends public interface of GenericStep for the truncated conjugate gradient method.
    template <class Cache, class Name>
    class InterfaceImpl : public CGSpec::InterfaceImpl<Cache,Name>
    {
    public:
      bool terminate() const
      {
        return cache_->doTerminate;
      }

      bool operatorIsPositiveDefinite() const
      {
        return cache_->operatorType == OperatorType::POSITIVE_DEFINITE;
      }

      void setPerformBlindUpdate(bool blindUpdate = true)
      {
        cache_->performBlindUpdate = blindUpdate;
      }

    protected:
      using CGSpec::InterfaceImpl<Cache,Name>::cache_;
    };


    /// Bind second template argument of TCG::InterfaceImpl to satisfy the interface of GenericStep.
    template < class Domain, class Range >
    using Interface = InterfaceImpl< Cache<Domain,Range>, Name >;

    /// Compute scaling of the search direction for the conjugate gradient method.
    struct Scaling : Mixin::Verbosity
    {
      template < class Cache >
      void operator()( Cache& cache ) const
      {
        if( cache.dxAdx > 0 )
        {
          cache.alpha = cache.sigma/cache.dxAdx;
          return;
        }

        if( verbosityLevel() > Mixin::Verbosity::BRIEF )
          std::cout << "    " << "Truncating at nonconvexity" << std::endl;
        // At least do something to retain a little chance to get out of the nonconvexity. If a nonconvexity is encountered in the first step something probably went wrong
        // elsewhere. Chances that a way out of the nonconvexity can be found are small in this case.
        if( cache.performBlindUpdate )
          cache.x += cache.dx;

        cache.alpha = 0;
        cache.doTerminate = true;
        cache.operatorType = OperatorType::INDEFINITE;
      }
    };


    /// Step implementation for the truncated conjugate gradient method.
    template <class Domain, class Range = Domain>
    using Step =
    GenericStep< Domain,
                 Range,
                 Interface< Domain, Range >,
                 CGSpec::ApplyPreconditioner,
                 CGSpec::SearchDirection,
                 Scaling,
                 CGSpec::Update >;
  }
  /// @endcond


  /*!
    @ingroup ISTL_Solvers
    @brief Truncated conjugate gradient method.

    Compute a descent direction for quadratic optimization problems of the
    form \f$ \frac{1}{2}x^T Ax - b^T x \f$, where \f$A:\ X\mapsto Y\f$ is a possibly indefinite linear operator.

    Terminates if a conjugate search direction \f$\delta x\f$ of non-positive curvature is encountered, i.e. if \f$\delta xA\delta x\le 0\f$.

    @tparam Domain domain space \f$X\f$
    @tparam Range range space \f$Y\f$
    @tparam TerminationCriterion termination criterion (such as Dune::KrylovTerminationCriterion::ResidualBased or Dune::KrylovTerminationCriterion::RelativeEnergyError (default))
   */
  template <class Domain, class Range = Domain,
            template <class> class TerminationCriterion = KrylovTerminationCriterion::RelativeEnergyError>
  using TCGSolver = GenericIterativeMethod< TCGSpec::Step<Domain,Range> , TerminationCriterion< real_t<Domain> > >;
}

#endif // DUNE_TCG_SOLVER_HH
