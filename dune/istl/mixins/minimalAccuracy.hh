#ifndef DUNE_MIXIN_MINIMAL_ACCURACY_HH
#define DUNE_MIXIN_MINIMAL_ACCURACY_HH

#include <cassert>

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for minimal accuracy.
     */
    template <class real_type=double>
    class MinimalAccuracy
    {
    public:
      /**
       * @brief Constructor.
       * @param accuracy minimal accuracy
       */
      explicit MinimalAccuracy(real_type accuracy = 0.25) noexcept
        : minimalAccuracy_{accuracy}
      {
        assert(minimalAccuracy_ >= 0);
      }

      /**
       * @brief Set minimal accuracy.
       * @param accuracy minimal accuracy
       */
      void setMinimalAccuracy(real_type accuracy)
      {
        assert(accuracy >= 0);
        minimalAccuracy_ = accuracy;
        connection_.update( minimalAccuracy() );
      }

      /**
       * @brief Access minimal accuracy.
       * @return minimal accuracy
       */
      real_type minimalAccuracy() const noexcept
      {
        return minimalAccuracy_;
      }

      /// Attach Impl::setMinimalAccuracy. The attached function will be called in setMinimalAccuracy().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setMinimalAccuracy, &impl, std::placeholders::_1) );
      }

    private:
      real_type minimalAccuracy_;
      Connection<real_type> connection_;
    };
  }
}

#endif // DUNE_MIXIN_MINIMAL_ACCURACY_HH
