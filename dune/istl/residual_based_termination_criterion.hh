#ifndef DUNE_RESIDUAL_BASED_TERMINATION_CRITERION_HH
#define DUNE_RESIDUAL_BASED_TERMINATION_CRITERION_HH

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <dune/common/timer.hh>
#include <dune/common/typetraits.hh>

#include "dune/istl/mixins/eps.hh"
#include "dune/istl/mixins/relativeAccuracy.hh"
#include "dune/istl/mixins/verbosity.hh"

namespace Dune
{
  /*! @cond */
  class InverseOperatorResult;
  /*! @endcond */

  /// Termination criteria for Krylov subspace methods.
  namespace KrylovTerminationCriterion
  {
    /*!
      @ingroup ISTL_Solvers
      @brief Residual-based relative error criterion.

      This termination criterion can only be used with step implementations that provide a member function
      @code
      real_type Step::residualNorm();
      @endcode
     */
    template <class real_type>
    class ResidualBased :
        public Mixin::Eps<real_type>,
        public Mixin::RelativeAccuracy<real_type>,
        public Mixin::Verbosity
    {
    public:
      /*!
        @brief Constructor.
        @param accuracy required relative accuracy of the residual
        @param eps maximal attainable accuracy
       */
      explicit ResidualBased(real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                             real_type eps = std::numeric_limits<real_type>::epsilon(),
                             unsigned verbosityLevel = 0)
        : Mixin::Eps<real_type>{eps},
          Mixin::RelativeAccuracy<real_type>{accuracy},
          Mixin::Verbosity{verbosityLevel}
      {}

      /// Reset/Initialize internal state before using the termination criterion.
      void reset()
      {
        assert(step_residualNorm_);
        initialResidualNorm_ = step_residualNorm_();
        iteration_ = 0;
        timer_.reset();
        timer_.start();
      }

      /*!
        @brief Connect to step implementation.

        @warning operator bool() seg-faults if no step is connected

        @param step step implementation that provides a member function step.residualNorm()
       */
      template <class Step,
                class = std::enable_if<std::is_reference<Step>::value> >
      void connect(Step&& step)
      {
        step_residualNorm_ = std::bind(&std::decay<Step>::type::residualNorm,&step);
      }

      /*!
        @brief Write information to res.
        @param res holds information on required iterations, reduction, convergence, rate and elapsed time.
       */
      void write(InverseOperatorResult& res)
      {
        assert(step_residualNorm_);
        res.iterations = iteration_;
        res.reduction = step_residualNorm_()/initialResidualNorm_;
        res.conv_rate = pow(res.reduction,1./res.iterations);
        res.elapsed = timer_.stop();
      }

      /*!
        @brief Evaluate termination criterion.
        @return true if termination criterion is satisfied, else false
       */
      operator bool()
      {
        ++iteration_;

        auto acc = std::max(this->eps(),this->relativeAccuracy());

        if( verbosityLevel() > 1 )
          std::cout << "Estimated error (res.-based): " << errorEstimate() << std::endl;

        if( errorEstimate() < acc )
          return true;

        return false;
      }

      /// Access relative residual error.
      real_type errorEstimate() const
      {
        assert( step_residualNorm_ );
        return step_residualNorm_()/initialResidualNorm_;
      }

    private:
      real_type initialResidualNorm_ = -1;
      unsigned iteration_ = 0;
      std::function<real_type()> step_residualNorm_;
      Timer timer_ = Timer{ false };
    };
  }
}

#endif // DUNE_RESIDUAL_BASED_TERMINATION_CRITERION_HH
