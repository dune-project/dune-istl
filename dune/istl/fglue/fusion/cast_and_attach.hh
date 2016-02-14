#ifndef DUNE_FGLUE_FUSION_CAST_AND_ATTACH_HH
#define DUNE_FGLUE_FUSION_CAST_AND_ATTACH_HH

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    namespace Fusion
    {
      template <class Source>
      struct CastAndAttach
      {
        CastAndAttach(Source& source)
          : source_(source)
        {}

        template <class Target>
        void operator()(Target& target)
        {
          source_.attach(static_cast<Source&>(target));
        }

      private:
        Source& source_;
      };


      template <class Source>
      struct CastAndDetach
      {
        CastAndDetach(Source& source)
          : source_(source)
        {}

        template <class Target>
        void operator()(Target& target)
        {
          source_.detach(static_cast<Source&>(target));
        }

      private:
        Source& source_;
      };
    }
  }
}

#endif // DUNE_FGLUE_FUSION_CAST_AND_ATTACH_HH
