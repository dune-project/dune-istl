#ifndef DUNE_TRCG_SOLVER_HH
#define DUNE_TRCG_SOLVER_HH

#include <functional>
#include <iostream>
#include <string>
#include <utility>

#include <dune/common/typetraits.hh>
#include "dune/istl/cg_solver.hh"
#include "dune/istl/generic_iterative_method.hh"
#include "dune/istl/generic_step.hh"
#include "dune/istl/rcg_solver.hh"
#include "dune/istl/relative_energy_termination_criterion.hh"

namespace Dune
{
  /// @cond
  namespace TRCGSpec
  {
    /// Cache object for the truncated regularized conjugate gradient method.
    template <class Domain, class Range>
    struct Cache : RCGSpec::Cache<Domain,Range>
    {
      template <class... Args>
      Cache(Args&&... args)
        : RCGSpec::Cache<Domain,Range>( std::forward<Args>(args)... )
      {}

      std::function<bool()> minimalDecreaseAchieved = {};
    };


    /// "Truncated Regularized Conjugate Gradients"
    class Name
    {
    public:
      std::string name() const
      {
        return "Truncated Regularized Conjugate Gradients";
      }
    };


    /// Extends public interface of GenericStep for the truncated regularized conjugate gradient method.
    template < class Cache_ >
    class InterfaceImpl : public RCGSpec::InterfaceImpl< Cache_, Name >
    {
    public:
      using Cache = Cache_;

      template <class... Args>
      InterfaceImpl(Args&&... args)
        : RCGSpec::InterfaceImpl<Cache,Name>( std::forward<Args>(args)... )
      {}

      void connect(std::function<bool()> minimalDecreaseAchieved)
      {
        cache_->minimalDecreaseAchieved = std::move(minimalDecreaseAchieved);
      }

      bool terminate() const
      {
        return cache_->doTerminate;
      }

    protected:
      using RCGSpec::InterfaceImpl<Cache,Name>::cache_;
    };

    template < class Domain, class Range >
    using Interface = InterfaceImpl< Cache<Domain,Range> >;


    /**
     * @copydoc CG::Scaling<real_type>
     *
     * Regularize or truncate at directions of non-positive curvature.
     */
    template <class real_type>
    class Scaling : public RCGSpec::Scaling<real_type>
    {
    public:
      template < class Cache >
      void operator()( Cache& cache )
      {
        if( cache.dxAdx > 0 )
        {
          cache.alpha = cache.sigma/cache.dxAdx;
          return;
        }

        assert(cache.minimalDecreaseAchieved);
        if( cache.minimalDecreaseAchieved() )
        {
          if( this->verbosityLevel() > Mixin::Verbosity::BRIEF )
            std::cout << "    " << "Truncating at nonconvexity." << std::endl;
          cache.alpha = 0;
          cache.operatorType = OperatorType::INDEFINITE;
          cache.doTerminate = true;
          return;
        }

        RCGSpec::Scaling<real_type>::operator()( cache );
      }
    };

    /// Step implementation for the truncated regularized conjugate gradient method.
    template <class Domain, class Range=Domain>
    using Step =
    GenericStep< Domain,
                 Range,
                 Interface< Domain, Range >,
                 CGSpec::ApplyPreconditioner,
                 RCGSpec::SearchDirection,
                 Scaling< real_t<Domain> >,
                 RCGSpec::Update >;
  }
  /// @endcond


  /*!
    @ingroup ISTL_Solvers
    @brief Truncated regularized conjugate gradient method (see @cite Lubkoll2015a).

    Compute a descent direction for quadratic optimization problems of the
    form \f$ q(x)=\frac{1}{2}x^T Ax - b^T x \f$, where \f$A:\ X\mapsto Y\f$ is a possibly indefinite linear operator.

    Combines the regularization strategy of RCG with TCG if a conjugate search direction \f$\delta x\f$ of non-positive curvature is encountered, i.e. if \f$\delta xA\delta x\le 0\f$.

    Suppose that \f$q\f$ is a local model of a nonconvex optimization problem. Far from the solution seach directions do not need to be computed overly accurate, say up to some relative
    accuracy \f$\delta_{min}\f$. Close to the solution we need to compute more accurate corrections, say up to some relative accuracy \f$\delta\f$. In this setting the TRCG method
    treats directions of non-positive curvature as follows:
     - If \f$\delta < \delta_{min}\f$ and the current iterate is acceptable with respect to \f$\delta_{min}\f$ then we conclude that we are still far from the solution, (where \f$A\f$ is positive definite)
       and accept the computed iterate.
     - Else the regularization strategy of RCG is applied.

    @tparam Domain domain space \f$X\f$
    @tparam Range range space \f$Y\f$
    @tparam TerminationCriterion termination criterion (such as Dune::KrylovTerminationCriterion::RelativeEnergyError, must provide a member function bool minimalDecreaseAchieved())
   */
  template <class Domain, class Range = Domain,
            template <class> class TerminationCriterion = KrylovTerminationCriterion::RelativeEnergyError>
  using TRCGSolver = GenericIterativeMethod< TRCGSpec::Step<Domain,Range> , TerminationCriterion< real_t<Domain> > >;
}

#endif // DUNE_TRCG_SOLVER_HH
