#ifndef DUNE_TERMINATION_CRITERIA_HH
#define DUNE_TERMINATION_CRITERIA_HH

#include <algorithm>
#include <cmath>
#include <limits>

#include <dune/common/timer.hh>
#include <dune/common/typetraits.hh>

#include "dune/istl/mixins/mixins.hh"

namespace Dune
{
  /*! @cond */
  class InverseOperatorResult;
  /*! @endcond */

  namespace KrylovTerminationCriterion
  {
    /*!
      @ingroup ISTL_Solvers
      @brief %Termination criterion for conjugate gradient methods based on an estimate of the relative energy error.

      Relative energy error termination criterion according to @cite Strakos2005 (see also @cite Hestenes1952, and for a related absolute energy error
      criterion @cite Arioli2004).

      Requires that CG starts at \f$ x = 0 \f$. More general starting values might be used, but must be chosen such that
      the estimate for the energy norm of the solution stays positive (see the above mentioned paper for details).

      The essential idea behind this termination criterion is simple: perform \f$d\f$ extra iterations of the conjugate gradient
      method to estimate the absolute or relative error in the energy norm (the parameter \f$d\f$ can be adjusted with setLookAhead()).
      To compute the error estimate only quantities that are anyway computed as intermediate results in the conjugate gradient method are required.

      This estimate only relies on local orthogonality and thus its evaluation is numerically stable.
     */
    template <class real_type>
    class RelativeEnergyError :
        public Mixin::AbsoluteAccuracy<real_type>,
        public Mixin::RelativeAccuracy<real_type>,
        public Mixin::MinimalAccuracy<real_type>,
        public Mixin::Eps<real_type>,
        public Mixin::Verbosity
    {
      /*! @cond */
      class Step
      {
        class AbstractBase
        {
        public:
          virtual ~AbstractBase(){}
          virtual real_type alpha() const = 0;
          virtual real_type length() const = 0;
          virtual real_type preconditionedResidualNorm() const = 0;
          virtual AbstractBase* clone() const = 0;
        };

        template <class Type>
        class Base : public AbstractBase
        {
        public:
          Base(const Type& type) : type_(&type) {}
          real_type alpha() const final override { return type_->alpha(); }
          real_type length() const final override { return type_->length(); }
          real_type preconditionedResidualNorm() const final override { return type_->preconditionedResidualNorm(); }
          Base* clone() const { return new Base{*type_}; }
        private:
          const Type* type_;
        };

      public:
        Step()
          : impl_(nullptr)
        {}

        Step(Step&&) = default;
        Step& operator=(Step&&) = default;

        Step(const Step& other)
          : impl_( other.impl_ == nullptr ? nullptr : other.impl_->clone() )
        {}

        Step& operator=(const Step& other)
        {
          impl_ = other.impl_ == nullptr ? nullptr : std::unique_ptr<AbstractBase>(other.impl_->clone());
        }

        template <class Type>
        Step& operator=(const Type& type)
        {
          impl_ = std::unique_ptr< Base<Type> >( new Base<Type>(type) );
          return *this;
        }

        template <class Type,
                  typename std::enable_if< std::is_rvalue_reference<Type>::value >::type* = nullptr >
        Step& operator=(Type&&)
        {
          throw std::invalid_argument("Step can not be assigned with a temporary object.");
          return *this;
        }

        real_type alpha() const { return impl_->alpha(); }

        real_type length() const { return impl_->length(); }

        real_type preconditionedResidualNorm() const { return impl_->preconditionedResidualNorm(); }

        operator bool() const { return impl_!=nullptr; }

      private:
        std::unique_ptr<AbstractBase> impl_ = nullptr;
      };
      /*! @endcond */
    public:
      /*!
        @brief Constructor.
        @param relativeAccuracy required relative accuracy for the estimated energy error
        @param verbosity verbosity level
        @param eps maximal attainable accuracy
        @param absoluteAccuracy absolute accuracy
       */
     explicit RelativeEnergyError(real_type relativeAccuracy = std::numeric_limits<real_type>::epsilon(),
                                  unsigned verbosity = 0,
                                  real_type eps = std::numeric_limits<real_type>::epsilon(),
                                  real_type absoluteAccuracy = std::numeric_limits<real_type>::epsilon())
        : Mixin::AbsoluteAccuracy<real_type>{absoluteAccuracy},
          Mixin::RelativeAccuracy<real_type>{relativeAccuracy},
          Mixin::Eps<real_type>{eps},
          Mixin::Verbosity{verbosity}
      {}

      //! @copydoc ResidualBased::operator bool()
      operator bool()
      {
        readParameter();

        if( verbosityLevel() > 1 )
          std::cout << "Estimated error (rel. energy error): " << errorEstimate() << std::endl;

        if( vanishingStep() )
          return true;

        using std::max;
        auto acc = max( this->relativeAccuracy() , this->eps() );
        return ( scaledGamma2_.size() > lookAhead_ && errorEstimate() < acc );
      }

      //! @copydoc ResidualBased::reset()
      void reset()
      {
        scaledGamma2_.clear();
        energyNorm2_ = stepLength2_ = 0;
        timer_.reset();
        timer_.start();
      }

      /// Access estimated relative energy error.
      real_type errorEstimate() const
      {
        return sqrt( squaredRelativeError() );
      }

      /*!
        @brief Set the additional CG-iterations required for estimating the relative energy error.

        @param lookAhead the requested lookahead value (default = 5)
       */
      void setLookAhead(unsigned lookAhead = 5)
      {
        lookAhead_ = lookAhead;
      }

      /*!
        @brief Relaxed termination criterion.
        @return true if the iteration has reached some minimal required accuracy, possibly bigger than the desired accuracy. Currently, this method is required in the
         truncated regularized conjugate gradient method only.
       */
      bool minimalDecreaseAchieved() const
      {
        return squaredRelativeError() < this->minimalAccuracy() * this->minimalAccuracy();
      }


      //! @copydoc ResidualBased::connect()
      template <class Step>
      void connect(Step&& step)
      {
        step_ = std::forward<Step>( step );
      }

      //! @copydoc ResidualBased::write()
      void write(InverseOperatorResult& res)
      {
        res.iterations = scaledGamma2_.size();
        res.reduction = errorEstimate();
        res.conv_rate = pow( res.reduction, 1./res.iterations );
        res.elapsed = timer_.stop();
      }

      /*!
        @brief check if the energy norm of the current step \f$\|q\|_A=\sqrt(qAq)\f$ is smaller than the maximal attainable accuracy multiplied with the energy norm of the iterate \f$\varepsilon_{max}\|x\|_A\f$.
        @return true if \f$\|q\|<\varepsilon_{max}\|x\|_A\f$, else false
       */
      bool vanishingStep() const
      {
        auto acc2 = this->absoluteAccuracy() * this->absoluteAccuracy();
        using std::min;
        if( energyNorm2_ > acc2 ) acc2 = min( acc2, this->eps() * this->eps() * energyNorm2_ );
        return stepLength2_ < acc2;
      }

    private:
      void readParameter()
      {
        assert( step_ );
        scaledGamma2_.push_back( step_.alpha() * step_.preconditionedResidualNorm() );
        energyNorm2_ += scaledGamma2_.back();
        using std::abs;
        stepLength2_ = abs( step_.length() );
      }

      real_type squaredRelativeError() const
      {
        if( scaledGamma2_.size() < lookAhead_ ) return std::numeric_limits<real_type>::max();
        return std::accumulate(scaledGamma2_.end() - lookAhead_, scaledGamma2_.end(), real_type(0)) / energyNorm2_;
      }

      unsigned lookAhead_ = 10;
      real_type energyNorm2_ = 0;
      real_type stepLength2_ = 0;
      std::vector<real_type> scaledGamma2_ = std::vector<real_type>{ };
      Step step_ = { };
      Timer timer_ = Timer{ false };
    };
  }
}

#endif // DUNE_TERMINATION_CRITERIA_HH
