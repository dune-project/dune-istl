#ifndef DUNE_MIXIN_RELATIVE_ACCURACY_HH
#define DUNE_MIXIN_RELATIVE_ACCURACY_HH

#include <cassert>
#include <limits>

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for relative accuracy.
     */
    template <class real_type=double>
    class RelativeAccuracy
    {
    public:
      /**
       * @brief Constructor.
       * @param accuracy relative accuracy.
       */
      explicit RelativeAccuracy(real_type accuracy = std::numeric_limits<real_type>::epsilon()) noexcept
        : relativeAccuracy_{accuracy}
      {
        assert(relativeAccuracy_ >= 0);
      }

      /**
       * @brief Set relative accuracy.
       * @param accuracy relative accuracy
       */
      void setRelativeAccuracy(real_type accuracy)
      {
        assert(accuracy>= 0);
        relativeAccuracy_ = accuracy;
        connection_.update( relativeAccuracy() );
      }

      /**
       * @brief Access relative accuracy.
       * @return relative accuracy
       */
      real_type relativeAccuracy() const noexcept
      {
        return relativeAccuracy_;
      }

      /// Attach Impl::setRelativeAccuracy. The attached function will be called in setRelativeAccuracy().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setRelativeAccuracy, &impl, std::placeholders::_1) );
      }

    private:
      real_type relativeAccuracy_;
      Connection<real_type> connection_;
    };
  }
}

#endif // DUNE_MIXIN_RELATIVE_ACCURACY_HH
