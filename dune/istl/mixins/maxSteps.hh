#ifndef DUNE_MIXIN_MAX_STEPS_HH
#define DUNE_MIXIN_MAX_STEPS_HH

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for maximal number of steps/iterations.
     */
    class MaxSteps
    {
    public:
      /**
       * @brief Constructor.
       * @param maxSteps maximal number of steps/iterations
       */
      explicit MaxSteps(unsigned maxSteps = 100) noexcept
        : maxSteps_{maxSteps}
      {}

      /**
       * @brief Set maximal number of steps/iterations for iterative solvers.
       * @param maxSteps maximal number of steps/iterations
       */
      void setMaxSteps(unsigned maxSteps)
      {
        maxSteps_ = maxSteps;
        connection_.update( maxSteps_ );
      }

      /**
       * @brief Get maximal number of steps/iterations for iterative solvers.
       * @return maximal number of steps/iterations
       */
      unsigned maxSteps() const noexcept
      {
        return maxSteps_;
      }

      /// Attach Impl::setMaxSteps. The attached function will be called in setMaxSteps().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setMaxSteps, &impl, std::placeholders::_1) );
      }

    private:
      unsigned maxSteps_;
      Connection<unsigned> connection_;
    };
  }
}

#endif // DUNE_MIXIN_MAX_STEPS_HH
