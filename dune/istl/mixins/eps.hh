#ifndef DUNE_MIXIN_EPS_HH
#define DUNE_MIXIN_EPS_HH

#include <cassert>
#include <cmath>
#include <limits>

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for maximal attainable accuracy \f$\varepsilon\f$.
     */
    template <class real_type=double>
    class Eps
    {
    public:
      /**
       * @brief Constructor.
       * @param eps maximal attainable accuracy \f$\varepsilon\f$
       */
      explicit Eps( real_type eps = std::numeric_limits<real_type>::epsilon() ) noexcept
        : eps_{eps}
      {
        assert(eps_ > 0);
      }

      /**
       * @brief Set maximal attainable accuracy \f$\varepsilon\f$.
       * @param eps new maximal attainable accuracy
       */
      void setEps(real_type eps)
      {
        assert(eps > 0);
        eps_ = eps;
        connection_.update( eps_ );
      }

      /**
       * @brief Access maximal attainable accuracy.
       * @return \f$\varepsilon\f$
       */
      real_type eps() const noexcept
      {
        return eps_;
      }

      /**
       * @brief Access square root of maximal attainable accuracy.
       * @return \f$\sqrt\varepsilon\f$
       */
      real_type sqrtEps() const
      {
        return sqrt(eps_);
      }

      /**
       * @brief Get third root of maximal attainable accuracy.
       * @return \f$\varepsilon^{1/3}\f$
       */
      real_type cbrtEps() const
      {
        return cbrt(eps_);
      }

      /// Attach Impl::setEps. The attached function will be called in setEps().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setEps, &impl, std::placeholders::_1) );
      }

    private:
      real_type eps_;
      Connection<real_type> connection_;
    };
  }
}

#endif // DUNE_MIXIN_EPS_HH
