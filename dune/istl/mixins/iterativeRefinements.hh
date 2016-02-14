#ifndef DUNE_MIXIN_ITERATIVE_REFINEMENTS_HH
#define DUNE_MIXIN_ITERATIVE_REFINEMENTS_HH

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for iterative refinements.
     */
    class IterativeRefinements
    {
    public:
      /**
       * @brief Constructor.
       * @param refinements number of iterative refinements.
       */
      explicit IterativeRefinements(unsigned refinements = 0) noexcept
        : iterativeRefinements_{refinements}
      {}

      /**
       * @brief Set number of iterative refinements.
       * @param refinements number of iterative refinements
       */
      void setIterativeRefinements(unsigned refinements)
      {
        iterativeRefinements_ = refinements;
        connection_.update( iterativeRefinements() );
      }

      /**
       * @brief Access number of iterative refinements.
       * @return number of iterative refinements
       */
      unsigned iterativeRefinements() const noexcept
      {
        return iterativeRefinements_;
      }

      /// Attach Impl::setIterativeRefinements. The attached function will be called in setIterativeRefinements().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setIterativeRefinements, &impl, std::placeholders::_1) );
      }

    private:
      unsigned iterativeRefinements_;
      Connection<unsigned> connection_;
    };
  }
}
#endif // DUNE_MIXIN_ITERATIVE_REFINEMENTS_HH
