#ifndef DUNE_RCG_SOLVER_HH
#define DUNE_RCG_SOLVER_HH

#include <limits>
#include <utility>

#include <dune/common/typetraits.hh>
#include "dune/istl/cg_solver.hh"
#include "dune/istl/generic_iterative_method.hh"
#include "dune/istl/generic_step.hh"
#include "dune/istl/operator_type.hh"
#include "dune/istl/relative_energy_termination_criterion.hh"
#include "dune/istl/tcg_solver.hh"
#include "dune/istl/mixins/verbosity.hh"

namespace Dune
{
  /// @cond
  namespace RCGSpec
  {
    /// Cache object for the regularized conjugate gradient method.
    template <class Domain, class Range>
    struct Cache : TCGSpec::Cache<Domain,Range>
    {
      template <class... Args>
      Cache(Args&&... args)
        : TCGSpec::Cache<Domain,Range>( std::forward<Args>(args)... ),
          Pdx( this->r )
      {}

      /// @copydoc CGSpect::Cache::reset()
      void reset(LinearOperator<Domain,Range>* A,
                Preconditioner<Domain,Range>* P,
                ScalarProduct<Domain>* sp)
      {
        TCGSpec::Cache<Domain,Range>::reset(A,P,sp);
        Pdx = this->r;
        doRestart = false;
      }

      real_t<Domain> theta = 0, dxPdx = 0, minIncrease = 2, maxIncrease = 1000;
      Range Pdx;
      bool doRestart = false;
    };


    /**
     * @brief Extends public interface of GenericStep for the regularized conjugate gradient method.
     * @warning Interface is only valid as long as a suitable cache object is provided.
     */
    template <class Cache, class Name>
    class InterfaceImpl : public TCGSpec::InterfaceImpl<Cache,Name>
    {
    public:
      /// Restart the regularized conjugate gradient method after regularization.
      bool restart() const
      {
        return cache_->doRestart;
      }

      /**
       * @brief Set minimal ratio for increasing the regularization parameter, i.e. \f$\frac{\theta_{new}}{\theta_{old}}>=minIncrease\f$.
       * @param minIncrease minimal ratio for increasing the regularization parameter
       */
      template <class Type>
      void setMinimalIncrease(Type minIncrease)
      {
        cache_->minIncrease = minIncrease;
      }

      /**
       * @brief Set maximal ratio for increasing the regularization parameter, i.e. \f$\frac{\theta_{new}}{\theta_{old}}<=maxIncrease\f$.
       * @param maxIncrease maximal ratio for increasing the regularization parameter
       */
      template <class Type>
      void setMaximalIncrease(Type maxIncrease)
      {
        cache_->maxIncrease = maxIncrease;
      }

    protected:
      using TCGSpec::InterfaceImpl<Cache,Name>::cache_;
    };

    /// "Regularized Conjugate Gradients"
    class Name
    {
    public:
      std::string name() const
      {
        return "Regularized Conjugate Gradients";
      }
    };

    /// Bind second template argument of RCG::InterfaceImpl to satisfy the interface of GenericStep.
    template < class Domain, class Range >
    using Interface = InterfaceImpl< Cache<Domain,Range>, Name >;


    /// Compute search direction for the regularized conjugate gradient method.
    class SearchDirection : public CGSpec::SearchDirection
    {
    public:
      template <class Cache>
      void operator()( Cache& cache ) const
      {
        CGSpec::SearchDirection::operator()( cache );

        // adjust energy norm of correction
        cache.dxPdx = cache.sp->dot(cache.dx,cache.Pdx);
        cache.dxAdx += cache.theta * cache.dxPdx;
        // adjust preconditioned correction
        cache.Pdx *= cache.beta;
        cache.Pdx += cache.r;
      }
    };


    /// @copydoc CGSpec::Update
    class Update : public CGSpec::Update
    {
    public:
      template < class Cache >
      void operator()( Cache& cache ) const
      {
        CGSpec::Update::operator()( cache );
        cache.r.axpy(-cache.alpha*cache.theta,cache.Pdx);
      }
    };


    /**
     * @copydoc CGSpec::Scaling
     *
     * Triggers regularization and restart if a direction of non-positive curvature is encountered.
     */
    template <class real_type>
    class Scaling :
        public Mixin::Eps<real_type>,
        public Mixin::Verbosity
    {
    public:
      template < class Cache >
      void operator()( Cache& cache ) const
      {
        if( cache.dxAdx > 0 )
        {
          cache.alpha = cache.sigma/cache.dxAdx;
          return;
        }

        if( verbosityLevel() > Mixin::Verbosity::BRIEF )
          std::cout << "    Regularizing at nonconvexity: " << cache.dxAdx << std::endl;
        auto oldTheta = cache.theta > 0 ? cache.theta : this->eps();
        using std::abs;
        cache.theta += (1-cache.dxAdx)/abs(cache.dxPdx);
        using std::min;
        using std::max;
        cache.theta = min(max(cache.minIncrease*oldTheta,cache.theta),cache.maxIncrease*oldTheta);
        if( verbosityLevel() > Mixin::Verbosity::BRIEF )
          std::cout << "Updating regularization parameter from " << oldTheta << " to " << cache.theta << std::endl;

        cache.alpha = 0;
        cache.operatorType = OperatorType::INDEFINITE;
        cache.doRestart = true;
      }
    };


    //! Step implementation for the regularized conjugate gradient method.
    template <class Domain, class Range=Domain>
    using Step =
    GenericStep< Domain,
                 Range,
                 Interface< Domain, Range >,
                 CGSpec::ApplyPreconditioner,
                 SearchDirection,
                 Scaling< real_t<Domain> >,
                 Update >;
  }
  /// @endcond


  /*!
    @ingroup ISTL_Solvers
    @brief Regularized conjugate gradient method.

    Compute a descent direction for quadratic optimization problems of the
    form \f$ \frac{1}{2}x^T Ax - b^T x \f$, where \f$A:\ X\mapsto Y\f$ is a possibly indefinite linear operator.

    Regularizes if a conjugate search direction \f$\delta x\f$ of non-positive curvature is encountered, i.e. if \f$\delta xA\delta x\le 0\f$.
    In this case the operator \f$A\f$ is replaced by the operator \f$A+\theta P\f$, where \f$\theta\f$ is a monotone increasing regularization parameter.

    @tparam Domain domain space \f$X\f$
    @tparam Range range space \f$Y\f$
    @tparam TerminationCriterion termination criterion (such as Dune::KrylovTerminationCriterion::ResidualBased or Dune::KrylovTerminationCriterion::RelativeEnergyError (default))
   */
  template <class Domain, class Range = Domain,
            template <class> class TerminationCriterion = KrylovTerminationCriterion::RelativeEnergyError>
  using RCGSolver = GenericIterativeMethod< RCGSpec::Step<Domain,Range> , TerminationCriterion< real_t<Domain> > >;
}

#endif // DUNE_RCG_SOLVER_HH
