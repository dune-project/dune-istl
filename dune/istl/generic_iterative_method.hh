#ifndef DUNE_GENERIC_ITERATIVE_METHOD_HH
#define DUNE_GENERIC_ITERATIVE_METHOD_HH

#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <utility>

#include "dune/common/typetraits.hh"
#include "dune/istl/solver.hh"
#include "dune/istl/optional.hh"
#include "dune/istl/mixins/mixins.hh"
#include "dune/istl/fglue/tmp/create_missing_base_classes.hh"
#include "dune/istl/is_valid.hh"

namespace Dune
{
  //! @cond
  template <class,class> class LinearOperator;
  template <class,class> class Preconditioner;
  template <class> class ScalarProduct;

  namespace Detail
  {
    /// Empty default storage object.
    template <class domain_type, class range_type, class TerminationCriterion, class = void>
    struct Storage
    {};

    /**
     * @brief Storage object for GenericIterativeMethod with a termination criterion that may trigger restarts.
     *
     * Stores initial guess \f$ x0 \f$ and initial right hand side \f$ b0 \f$.
     */
    template <class domain_type, class range_type, class TerminationCriterion>
    struct Storage< domain_type, range_type, TerminationCriterion, void_t< Try::MemFn::restart<TerminationCriterion> > >
    {
      Storage(Storage&&) = default;
      Storage& operator=(Storage&&) = default;

      Storage(const Storage& other)
        : x0( other.x0 ? new domain_type(*other.x0) : nullptr ),
          b0( other.b0 ? new range_type(*other.b0) : nullptr)
      {}

      Storage& operator=(const Storage& other)
      {
        if( other.x0 )
          x0.reset( new domain_type(*other.x0) );
        if( other.b0 )
          b0.reset( new range_type(*other.b0) );
      }

      void init(const domain_type& x, const range_type& b)
      {
        x0.reset( new domain_type(x) );
        b0.reset( new range_type(b) );
      }

      void reset(domain_type& x, range_type& b) const
      {
        assert(x0 && b0);
        x = *x0;
        b = *b0;
      }

      std::unique_ptr<domain_type> x0;
      std::unique_ptr<range_type> b0;
    };


    using namespace FGlue;

    /// Is Empty if Step is derived from Mixin::Verbosity, else is Mixin::Verbosity.
    template <class Step>
    using EnableVerbosity = Apply< StoreIf< IsNotDerivedFrom<Step> > , Mixin::Verbosity >;

    /**
     * Generate a template meta-function that takes an arbitrary number of arguments and generates a
     * type that is derived from each argument that is a base class of TerminationCriterion, but not
     * of Step.
     */
    template <class Step,
              class TerminationCriterion>
    using AdditionalMixinsCondition =
    Apply< Delay<Or>,
      FGlue::IsBaseOf<TerminationCriterion>,
      FGlue::IsBaseOf<Step>
    >;

    /// Generate type that is derived from all necessary mixin base classes for GenericIterativeMethod.
    template <class Step,
              class TerminationCriterion,
              class real_type = real_t<typename Step::domain_type> >
    using AddMixins =
    Apply< Compose,
      EnableBaseClassesIf< AdditionalMixinsCondition<Step,TerminationCriterion> , DUNE_ISTL_MIXINS( real_type ) >,
      EnableVerbosity<Step>
    >;

    struct AlwaysFalse
    {
      operator bool() const
      {
        return false;
      }
    };
  }

  namespace Optional
  {
    template < class Step >
    using TryNestedType_Cache = typename Step::Cache;

    struct NoCache
    {
      template <class... Args>
      explicit NoCache(Args&&...) {}
    };

    template < class Step , class = void >
    struct StepTraits
    {
      using Cache = NoCache;

      static void setCache( const Step&, Cache* ) noexcept
      {}
    };

    template < class Step >
    struct StepTraits< Step, void_t< TryNestedType_Cache<Step> > >
    {
      using Cache = TryNestedType_Cache<Step>;

      static void setCache( Step& step, Cache* cache )
      {
        step.setCache(cache);
      }
    };

    template < class Step, class domain_type, class range_type >
    typename StepTraits< Step >::Cache createCache( domain_type& x, range_type& y )
    {
      return typename StepTraits< Step >::Cache( x, y );
    }

    template < class Step, class Cache >
    void setCache( Step& step, Cache* cache )
    {
      StepTraits< Step >::setCache( step, cache );
    }
  }
  //! @endcond


  /*!
    @ingroup ISTL_Solvers
    @brief Generic wrapper for iterative methods.

    Requires at least an implementation of one step of an iterative method.
    Optionally also a termination criterion can be specified.

    The GenericIterativeMethod provides parts of the interfaces of the step and the
    termination criterion. In particular all mixin classes, summarized in DUNE_ISTL_MIXINS,
    that are base classes of either the step or the termination criterion are also base classes
    of the generic iterative method. The corresponding setters will be automatically connected,
    i.e. the member function `%setRelativeAccuracy(real_type)` will set the relative accuracy
    of the termination criterion.

    @tparam Step_ implementation of one step of an iterative method
    @tparam TerminationCriterion_ termination criterion

    - The step implementation has to provide the following interface (member functions may also be const):
      - <b>Required:</b>
        @code
        class Step
        {
          using domain_type = ...;
          using range_type  = ...;
        public:
          // Move constructor.
          Step(Step&&);

          // Move assignment.
          Step& operator=(Step&&);

          // Perform one step of an iterative method.
          void compute(domain_type& x, range_type& b);
        };
        @endcode

      - <b>%Optional:</b> each of the following functions will be used if present:
        @code
        class Step
        {
          ...

          using Cache = ...;

          // Return name of the method.
          std::string name();

          // Set pointer to cache, to reduce memory allocations or for usage with GenericStep.
          void setCache(Cache* cache);

          // Initialization phase (i.e. for computation of initial residual,...).
          void init(domain_type& x, range_type& b);

          // Restart method.
          bool restart() const;

          // Reset method after restart.
          void reset(domain_type& x, range_type& b);

          // Postprocess computed solution (i.e. apply `Preconditioner::post(x)`);
          void postProcess(domain_type& x);
        };
        @endcode

    - The termination criterion has to provide the following interface (member functions may also be const):
      - <b>Required:</b>
        @code
        class TerminationCriterion
        {
        public:
          /// Return true if iteration should terminate, else false.
          operator bool();
        };
        @endcode

      - <b>%Optional:</b>
        @code
        class TerminationCriterion
        {
          ...

          using real_type = ...;

          // Reset/Initialize termination criterion.
          void reset();

          // Connect step implementation to termination criterion.
          // This function is required to transfer information from
          // the step computation to the termination criterion.
          template <class Step>
          void connect(Step&& step);

          // Set relative accuracy.
          void setRelativeAccuracy(real_type accuracy);

          // Write information into result.
          void write(InverseOperatorResult& result);

          // Access estimated error. If the verbosity level is bigger than 1, i.e.
          // if information is printed in each iteration, then this function should
          // be provided to monitor local convergence rates.
          real_type errorEstimate();
        };
        @endcode
   */
  template < class Step_,
             class TerminationCriterion = Detail::AlwaysFalse >
  class GenericIterativeMethod :
      public InverseOperator<typename Step_::domain_type, typename Step_::range_type> ,
      public Mixin::MaxSteps ,
      public Detail::AddMixins<Step_, TerminationCriterion>
  {
  public:
    /// Step implementation.
    using Step = Step_;
    /// Type of the domain space.
    using domain_type = typename Step::domain_type;
    /// Type of the range space.
    using range_type  = typename Step::range_type;
    /// Underlying field type.
    using field_type  = field_t<domain_type>;
    /// Underlying real type.
    using real_type = real_t<domain_type>;

    /*!
      @brief Construct from given step implementation and termination criterions.
      @param step object implementing one step of an iterative scheme
      @param terminate termination criterion
      @param maxSteps
     */
    GenericIterativeMethod(Step step,
                           TerminationCriterion terminate,
                           real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                           unsigned maxSteps = std::numeric_limits<unsigned>::max(),
                           unsigned verbosityLevel = Mixin::Verbosity::SILENT )
      : Mixin::MaxSteps(maxSteps) ,
        impl_(std::move(step)) ,
        terminationCriterion_(std::move(terminate))
    {
      initializeConnections();
      Optional::setRelativeAccuracy(*this,accuracy);
      this->setVerbosityLevel(verbosityLevel);
    }

    /// Optional constructor for the case that the step has a constructor that takes three parameters of type Operator, Preconditioner and ScalarProduct.
    template <class Operator, class Preconditioner, class ScalarProduct,
              typename std::enable_if< std::is_constructible< Step, Operator, Preconditioner, ScalarProduct >::value &&
                                       IsValidOperator< Operator, domain_type, range_type >::value &&
                                       IsValidPreconditioner< Preconditioner, domain_type, range_type >::value &&
                                       IsValidScalarProduct< ScalarProduct, domain_type >::value >::type* = nullptr>
    GenericIterativeMethod( Operator&& A,
                            Preconditioner&& P,
                            ScalarProduct&& sp,
                            TerminationCriterion terminate,
                            real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                            unsigned maxSteps = std::numeric_limits<unsigned>::max(),
                            unsigned verbosityLevel = Mixin::Verbosity::SILENT )
      : GenericIterativeMethod( Step( std::forward<Operator>(A), std::forward<Preconditioner>(P), std::forward<ScalarProduct>(sp) ) ,
                                std::move(terminate),
                                accuracy,
                                maxSteps,
                                verbosityLevel )
    {}

    /// Optional constructor for the case that the step has a constructor that takes three parameters of type Operator, Preconditioner and ScalarProduct and the termination criterion is default constructible.
    template <class Operator, class Preconditioner, class ScalarProduct,
              typename std::enable_if< std::is_constructible< Step, Operator, Preconditioner, ScalarProduct >::value &&
                                       std::is_default_constructible< TerminationCriterion >::value &&
                                       IsValidOperator< Operator, domain_type, range_type >::value &&
                                       IsValidPreconditioner< Preconditioner, domain_type, range_type >::value &&
                                       IsValidScalarProduct< ScalarProduct, domain_type >::value >::type* = nullptr>
    GenericIterativeMethod( Operator&& A,
                            Preconditioner&& P,
                            ScalarProduct&& sp,
                            real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                            unsigned maxSteps = std::numeric_limits<unsigned>::max(),
                            unsigned verbosityLevel = Mixin::Verbosity::SILENT )
      : GenericIterativeMethod( Step( std::forward<Operator>(A), std::forward<Preconditioner>(P), std::forward<ScalarProduct>(sp) ),
                                TerminationCriterion(),
                                accuracy,
                                maxSteps,
                                verbosityLevel )
    {}

    /// Optional constructor for the case that the step has a constructor that takes three parameters of type Operator and Preconditioner.
    template <class Operator, class Preconditioner,
              typename std::enable_if< std::is_constructible< Step, Operator, Preconditioner >::value &&
                                       IsValidOperator< Operator, domain_type, range_type >::value &&
                                       IsValidPreconditioner< Preconditioner, domain_type, range_type >::value >::type* = nullptr>
    GenericIterativeMethod( Operator&& A,
                            Preconditioner&& P,
                            TerminationCriterion terminate,
                            real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                            unsigned maxSteps = std::numeric_limits<unsigned>::max(),
                            unsigned verbosityLevel = Mixin::Verbosity::SILENT )
      : GenericIterativeMethod( Step( std::forward<Operator>(A), std::forward<Preconditioner>(P) ),
                                std::move(terminate),
                                accuracy,
                                maxSteps,
                                verbosityLevel )
    {}

    /// Optional constructor for the case that the step has a constructor that takes two parameters of type Operator and Preconditioner and the termination criterion is default constructible.
    template <class Operator, class Preconditioner,
              typename std::enable_if< std::is_constructible< Step, Operator, Preconditioner >::value &&
                                       std::is_default_constructible< TerminationCriterion >::value &&
                                       IsValidOperator< Operator, domain_type, range_type >::value &&
                                       IsValidPreconditioner< Preconditioner, domain_type, range_type >::value >::type* = nullptr>
    GenericIterativeMethod( Operator&& A,
                            Preconditioner&& P,
                            real_type accuracy = std::numeric_limits<real_type>::epsilon(),
                            unsigned maxSteps = std::numeric_limits<unsigned>::max(),
                            unsigned verbosityLevel = Mixin::Verbosity::SILENT )
      : GenericIterativeMethod( Step( std::forward<Operator>(A), std::forward<Preconditioner>(P) ),
                                TerminationCriterion(),
                                accuracy,
                                maxSteps,
                                verbosityLevel )
    {}

    /// Copy constructor.
    GenericIterativeMethod(const GenericIterativeMethod& other)
      : Mixin::MaxSteps(other),
        impl_(other),
        terminationCriterion_(other.terminationCriterion_)
    {
      initializeConnections();
    }

    /// Move constructor.
    GenericIterativeMethod(GenericIterativeMethod&& other)
      : Mixin::MaxSteps( other.maxSteps() ),
        impl_( std::move( other.impl_ ) ),
        terminationCriterion_( std::move( other.terminationCriterion_ ) )
    {
      initializeConnections();
    }

    /// Since references to operator, preconditioner and possibly scalar product are stored, copy assignment is not possible.
    GenericIterativeMethod& operator=(const GenericIterativeMethod&) = delete;

    /// Move assignment.
    GenericIterativeMethod& operator=(GenericIterativeMethod&& other)
    {
      impl_ = std::move(static_cast<Step&&>(other));
      Mixin::MaxSteps::operator=(std::move(other));
      terminationCriterion_ = std::move(other.terminationCriterion_);
      initializeConnections();
    }

    /*!
      @brief Apply iterative method to solve \f$Ax=b\f$.
      @param x initial iterate
      @param b initial right hand side
      @param res some statistics
     */
    void apply(domain_type& x, range_type& b, InverseOperatorResult& res) override
    {
      if( this->verbosityLevel() > Mixin::Verbosity::BRIEF )
        std::cout << "\n === " << Optional::name( impl_ ) << " === " << std::endl;

      auto cache = Optional::createCache< Step >( x, b );
      Optional::setCache( impl_, &cache );

      initialize(x,b);
      real_type lastErrorEstimate = -1;

      auto step=1u;
      for(; step<=maxSteps(); ++step)
      {
        impl_.compute(x,b);

        if( terminationCriterion_ )
          break;

        if( Optional::restart( impl_ ) )
        {
          Optional::reset(storage_,x,b);
          Optional::reset(impl_,x,b);
          Optional::reset(terminationCriterion_);
          step = 0u;
          lastErrorEstimate = -1;
          continue;
        }

        if( this->verbosityLevel() > Mixin::Verbosity::BRIEF )
        {
          lastErrorEstimate = Optional::errorEstimate( terminationCriterion_ );
          printOutput(step,lastErrorEstimate);
        }
      } // end iteration

      Optional::postProcess(impl_,x);
      if( step - 1 < maxSteps() ) res.converged = true;
      else res.converged = false;
      Optional::write(terminationCriterion_,res);
      if( this->is_verbose() ) printFinalOutput(res,step);
    }

    /*!
      @brief Apply iterative method to solve \f$Ax=b\f$.
      @param x initial iterate
      @param b initial right hand side
      @param relativeAccuracy required relative accuracy
      @param res some statistics
     */
    void apply( domain_type &x, range_type &b, double relativeAccuracy, InverseOperatorResult &res ) override
    {
      Optional::setRelativeAccuracy( terminationCriterion_, relativeAccuracy );
      apply( x, b, res );
    }

    /*!
      @brief Apply iterative method to solve \f$Ax=b\f$.
      @param x initial iterate
      @param b initial right hand side
     */
    void apply( domain_type &x, range_type &b )
    {
      InverseOperatorResult res;
      apply( x, b, res);
    }

    /// Access termination criterion.
    TerminationCriterion& getTerminationCriterion()
    {
      return terminationCriterion_;
    }

    /// Access step implementation.
    Step& getStep()
    {
      return impl_;
    }

  private:
    /// Initialize connections between iterative method and termination criterion.
    void initializeConnections()
    {
      // connect termination criterion to step implementation to access relevant data
      Optional::connect(terminationCriterion_,impl_);

      // attach mixins to correctly forward parameters to the termination criterion
      using namespace Mixin;
      Optional::Mixin::Attach< DUNE_ISTL_MIXINS( real_type ) >::apply( *this, impl_ );
      Optional::Mixin::Attach< DUNE_ISTL_MIXINS( real_type ) >::apply( *this, terminationCriterion_ );
    }

    void initialize(domain_type& x, range_type& b)
    {
      Optional::init(storage_,x,b);
      Optional::init(impl_,x,b);
      Optional::reset(terminationCriterion_);
    }

    void printOutput( unsigned step, real_type lastErrorEstimate ) const
    {
      this->printHeader( std::cout );
      InverseOperator< domain_type, range_type >::printOutput( std::cout,
                                                               static_cast<decltype( Optional::errorEstimate( terminationCriterion_ ) )>(step),
                                                               Optional::errorEstimate( terminationCriterion_ ),
                                                               lastErrorEstimate );
    }

    void printFinalOutput(const InverseOperatorResult& res, unsigned step) const
    {
      auto name = Optional::name( impl_ );
      name += (step==maxSteps()+1) ? ": Failed" : ": Converged";
      std::cout << "\n === " << name << " === " << std::endl;
      this->printHeader(std::cout);
      InverseOperator<domain_type,range_type>::printOutput(std::cout,
                                                           static_cast<double>(res.iterations),
                                                           res.reduction,
                                                           res.reduction/res.conv_rate);
      name = std::string(name.size(),'=');
      std::cout << " === " << name << "" << " === \n" << std::endl;
    }

    Step impl_;
    TerminationCriterion terminationCriterion_;
    Detail::Storage<domain_type,range_type,TerminationCriterion> storage_;
  };

  /*!
    @ingroup ISTL_Solvers
    @brief Generating function for GenericIterativeMethod.

    @param step implementation of one step of an iterative method
    @param terminationCriterion implementation of a termination criterion
   */
  template <class Step, class TerminationCriterion>
  GenericIterativeMethod< typename std::decay<Step>::type, typename std::decay<TerminationCriterion>::type >
  makeGenericIterativeMethod(Step&& step, TerminationCriterion&& terminationCriterion)
  {
    return GenericIterativeMethod< typename std::decay<Step>::type, typename std::decay<TerminationCriterion>::type >
        ( std::forward<Step>(step),
          std::forward<TerminationCriterion>(terminationCriterion) );
  }
}

#endif // DUNE_GENERIC_ITERATIVE_METHOD_HH
