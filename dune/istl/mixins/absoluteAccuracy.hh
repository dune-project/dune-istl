#ifndef DUNE_MIXIN_ABSOLUTE_ACCURACY_HH
#define DUNE_MIXIN_ABSOLUTE_ACCURACY_HH

#include <cassert>
#include <limits>

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for absolute accuracy.
     */
    template <class real_type=double>
    class AbsoluteAccuracy
    {
    public:
      /**
       * @brief Constructor.
       * @param accuracy absolute accuracy
       */
      explicit AbsoluteAccuracy(real_type accuracy = std::numeric_limits<real_type>::epsilon()) noexcept
        : absoluteAccuracy_{accuracy}
      {
        assert(absoluteAccuracy_ >= 0);
      }

      /**
       * @brief Set absolute accuracy.
       * @param accuracy absolute accuracy
       */
      void setAbsoluteAccuracy(real_type accuracy)
      {
        assert(accuracy >= 0);
        absoluteAccuracy_ = accuracy;
        connection_.update( absoluteAccuracy() );
      }

      /**
       * @brief Access absolute accuracy.
       * @return absolute accuracy
       */
      real_type absoluteAccuracy() const noexcept
      {
        return absoluteAccuracy_;
      }

      /// Attach Impl::setAbsoluteAccuracy. The attached function will be called in setAbsoluteAccuracy().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setAbsoluteAccuracy, &impl, std::placeholders::_1) );
      }

    private:
      real_type absoluteAccuracy_;
      Connection<real_type> connection_;
    };
  }
}

#endif // DUNE_MIXIN_ABSOLUTE_ACCURACY_HH
