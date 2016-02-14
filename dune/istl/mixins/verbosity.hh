#ifndef DUNE_MIXIN_VERBOSITY_HH
#define DUNE_MIXIN_VERBOSITY_HH

#include "connection.hh"

namespace Dune
{
  namespace Mixin
  {
    /**
     * @ingroup MixinGroup
     * @brief %Mixin class for verbosity.
     */
    class Verbosity
    {
    public:
      /// Predefined verbosity levels.
      enum { SILENT, BRIEF, DETAILED };

      /**
       * @brief Constructor.
       * @param verbosityLevel verbosity level
       */
      explicit Verbosity(unsigned verbosityLevel = 0) noexcept
        : verbosityLevel_{verbosityLevel}
      {}

      /**
       * @brief Enable/disable verbosity.
       * @param verbose true: if verbosityLevel = SILENT, set verbosityLevel = BRIEF; false: set verbosityLevel = SILENT
       */
      void setVerbosity(bool verbose)
      {
        if( verbose && verbosityLevel_ == SILENT )
          verbosityLevel_ = BRIEF;
        if( !verbose )
          verbosityLevel_ = SILENT;
        connection_.update( verbosityLevel() );
      }

      /**
       * @brief Check if verbosity is turned on.
       * @return true if verbosityLevel != SILENT
       */
      bool is_verbose() const noexcept
      {
        return verbosityLevel_ != SILENT;
      }

      /**
       * @brief Set verbosity level.
       * @param level verbosity level
       */
      void setVerbosityLevel(unsigned level)
      {
        verbosityLevel_ = level;
        connection_.update( verbosityLevel() );
      }

      /**
       * @brief Access verbosity level.
       * @return verbosity level
       */
      unsigned verbosityLevel() const noexcept
      {
        return verbosityLevel_;
      }

      /// Attach Impl::setVerbosityLevel. The attached function will be called in setVerbosityLevel().
      template <class Impl>
      void attach( Impl& impl )
      {
        connection_.attach( std::bind(&Impl::setVerbosityLevel, &impl, std::placeholders::_1) );
      }

    private:
      unsigned verbosityLevel_;
      Connection<unsigned> connection_;
    };
  }
}

#endif // DUNE_MIXIN_VERBOSITY_HH
