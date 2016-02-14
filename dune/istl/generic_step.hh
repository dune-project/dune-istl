#ifndef DUNE_GENERIC_STEP_HH
#define DUNE_GENERIC_STEP_HH

#include <string>
#include <utility>

#include <dune/common/typetraits.hh>
#include <dune/istl/scalarproducts.hh>

#include "dune/istl/mixins/mixins.hh"
#include "dune/istl/fglue/tmp/create_missing_base_classes.hh"
#include "dune/istl/fglue/fusion/connect.hh"
#include "dune/istl/is_valid.hh"

namespace Dune
{
  /// @cond
  template <class,class> class LinearOperator;
  template <class,class> class Preconditioner;

  namespace GenericStepDetail
  {
    class Ignore
    {
    public:
      template <class... Args>
      void operator()(Args&&...){}
    };


    template < class ApplyPreconditioner,
               class ComputeSearchDirection,
               class ComputeScaling,
               class Update,
               class real_type >
    using AddMixins =
    FGlue::EnableBaseClassesIf<
      FGlue::IsBaseOfOneOf< ApplyPreconditioner, ComputeSearchDirection, ComputeScaling, Update >,
      DUNE_ISTL_MIXINS( real_type )
    >;
  }
  /// @endcond

  /*!
    @ingroup ISTL_Solvers
    @brief Generic step of an iterative method.

    Solves a linear operator equation \f$Ax=b\f$ with \f$A:\ X\mapsto Y\f$, resp. one of its preconditioned versions
    \f$PAx=Px\f$ or \f$P_1AP_2P_2^{-1}x=P_1b\f$.

    @tparam Domain type of the domain space \f$X\f$
    @tparam Range type of the range space \f$Y\f$
    @tparam Interface base class for generic step for customization of the public interface, stores a pointer to a cache object
    @tparam ApplyPreconditioner functor specifying the application of the preconditioner, takes a reference to the cache object as argument
    @tparam ComputeSearchDirection functor specifying the computation of the search direction, takes a reference to the cache object as argument
    @tparam ComputeScaling functor specifying the computation of the step length parameter, takes a reference to the cache object as argument
    @tparam Update functor specifying the update of data, such as iterate, residual, ..., takes a reference to the cache object as argument

    The following steps are performed and may be specified:
      1. Apply preconditioner
      2. Compute search direction
      3. Compute scaling for the search direction (minimize wrt. search direction)
      4. Update data (such as iterate, residual, ...)

    The generic step is derived from `Interface` and all mixin classes that are base classes of at least of one of
    `ApplyPreconditioner, ComputeSearchDirection, ComputeScaling, Update`.

    - The class `Interface` must provide the following interface:
      @code
      class Interface
      {
      public:
        using Cache = ...

        void setCache(Cache* cache);

      protected:
        Cache* cache_;
      };
      @endcode

    - The class `ApplyPreconditioner` must provide the following interface:
      @code
      struct ApplyPreconditioner
      {
        template <class Preconditioner, class Domain, class Range>
        void pre(Preconditioner& P, Domain& x, Range& b);

        template <class Cache>
        void operator()(Cache& cache);

        template <class Preconditioner, class Domain>
        void post(Preconditioner& P, Domain& x);
      };
      @endcode

    - The classes `ComputeSearchDirection`, `ComputeScaling` and `Update` must provide the following interface:
      @code
      struct Other
      {
        template <class Cache>
        void operator()(Cache& cache);
      };
      @endcode
   */
  template <class Domain,
            class Range,
            class Interface,
            class ApplyPreconditioner    = GenericStepDetail::Ignore,
            class ComputeSearchDirection = GenericStepDetail::Ignore,
            class ComputeScaling         = GenericStepDetail::Ignore,
            class Update                 = GenericStepDetail::Ignore>
  class GenericStep :
      public GenericStepDetail::AddMixins< ApplyPreconditioner, ComputeSearchDirection, ComputeScaling, Update, real_t<Domain> >,
      public Interface
  {
  public:
    //! type of the domain space
    using domain_type = Domain;
    //! type of the range space
    using range_type = Range;
    //! underlying field type
    using field_type = field_t<Domain>;
    //! corresponding real type (same as real type for real spaces, differs for complex spaces)
    using real_type = real_t<Domain>;
    //! cache object storing temporaries
    using Cache = typename Interface::Cache;

    /**
     * @brief Constructor.
     * @param A linear operator
     * @param P preconditioner
     * @param sp scalar product
     */
    template <class Operator, class Preconditioner, class ScalarProduct,
              typename std::enable_if< IsValidOperator<Operator,domain_type,range_type>::value &&
                                       IsValidPreconditioner<Preconditioner,domain_type,range_type>::value &&
                                       IsValidScalarProduct<ScalarProduct,domain_type>::value>::type* = nullptr>
    GenericStep(Operator& A, Preconditioner& P, ScalarProduct& sp)
      : A_(A), P_(P), ssp_(), sp_(sp)
    {
      initializeConnections();
      static_assert( Operator::category == Preconditioner::category , "Linear operator and preconditioner are required to belong to the same category!" );
      static_assert( Operator::category == SolverCategory::sequential , "Linear operator must be sequential!" );
    }

    /**
     * @brief Constructor.
     * @param A linear operator
     * @param P preconditioner
     */
    template <class Operator, class Preconditioner,
              typename std::enable_if< IsValidOperator<Operator,domain_type,range_type>::value &&
                                       IsValidPreconditioner<Preconditioner,domain_type,range_type>::value>::type* = nullptr>
    GenericStep(Operator& A,  Preconditioner& P)
      : A_(A), P_(P), ssp_(), sp_(ssp_)
    {
      initializeConnections();
      static_assert( Operator::category == Preconditioner::category , "Linear operator and preconditioner are required to belong to the same category!" );
      static_assert( Operator::category == SolverCategory::sequential , "Linear operator must be sequential!" );
    }

    /// Copy constructor.
    GenericStep( const GenericStep& other )
      : A_( other.A_ ),
        P_( other.P_ ),
        ssp_( ),
        sp_( dynamic_cast< const SeqScalarProduct<domain_type>* >(&other.sp_) == nullptr ? other.sp_ : ssp_ )
    {
      initializeConnections( );
    }

    /// Move constructor.
    GenericStep( GenericStep&& other )
      : A_( other.A_ ),
        P_( other.P_ ),
        ssp_(),
        sp_( dynamic_cast< const SeqScalarProduct<domain_type>* >(&other.sp_) == nullptr ? other.sp_ : ssp_ )
    {
      initializeConnections( );
    }

    /*!
      @brief Apply preprocessing of the preconditioner.
      @param x initial iterate
      @param b initial right hand side
     */
    void init(domain_type& x, range_type& b)
    {
      applyPreconditioner_.pre( P_, x, b );
    }

    /*!
       @brief Reset cache.
       @param x iterate
       @param b right hand side
     */
    void reset(domain_type& x, range_type& b)
    {
      this->cache_->reset( &A_, &P_, &sp_ );
    }

    /*!
       @brief Set cache.
       @warning A cache must be set before using the GenericStep. Else no local storage is available.
     */
    void setCache( Cache* cache )
    {
      Interface::setCache( cache );
      this->cache_->reset( &A_, &P_, &sp_ );
    }

    /*!
      @brief Perform one step of an iterative method.
      @param x current iterate
      @param b current right hand side
     */
    void compute(domain_type&, range_type&)
    {
      assert( this->cache_ != nullptr );
      applyPreconditioner_( *this->cache_ );
      computeSearchDirection_( *this->cache_ );
      computeScaling_( *this->cache_ );
      update_( *this->cache_ );
    }



    /*!
      @brief Apply post-processing of the preconditioner, i.e. `P_.post(x)`.
      @param x final iterate
     */
    void postProcess(domain_type& x)
    {
      applyPreconditioner_.post(P_,x);
    }

  private:
    void initializeConnections()
    {
      using FGlue::Connector;
      using FGlue::IsDerivedFrom;

      Connector< IsDerivedFrom<Mixin::IterativeRefinements> >::template
          from< Mixin::IterativeRefinements >( *this ).
          to(applyPreconditioner_,computeSearchDirection_,computeScaling_,update_);
      Connector< IsDerivedFrom<Mixin::Verbosity> >::template
          from< Mixin::Verbosity >( *this ).
          to(applyPreconditioner_,computeSearchDirection_,computeScaling_,update_);
      Connector< IsDerivedFrom< Mixin::Eps<real_type> > >::template
          from< Mixin::Eps<real_type> >( *this ).
          to(applyPreconditioner_,computeSearchDirection_,computeScaling_,update_);
    }


    LinearOperator<Domain,Range>& A_;
    Preconditioner<Domain,Range>& P_;
    SeqScalarProduct<Domain> ssp_;
    ScalarProduct<Domain>& sp_;

    ApplyPreconditioner applyPreconditioner_;
    ComputeSearchDirection computeSearchDirection_;
    ComputeScaling computeScaling_;
    Update update_;
  };
}

#endif // DUNE_GENERIC_STEP_HH
