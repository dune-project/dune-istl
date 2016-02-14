#ifndef DUNE_FGLUE_FUSION_CONNECT_HH
#define DUNE_FGLUE_FUSION_CONNECT_HH

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/true_false.hh"
#include "dune/istl/fglue/fusion/apply_if.hh"
#include "dune/istl/fglue/fusion/cast_and_attach.hh"
#include "dune/istl/fglue/fusion/variadic_functor.hh"

namespace Dune
{
  namespace FGlue
  {
    template <class SourceOperation, class TargetOperation = SourceOperation>
    struct Connector
    {
      template <class Base, class Source, bool valid = isTrue< Apply<SourceOperation,Source> >() >
      struct From
      {
        From(Source& source)
          : castAndAttach(source)
        {}

        template <class... Targets>
        void to(Targets&... targets)
        {
          castAndAttach(targets...);
        }

        Fusion::VariadicFunctor< Fusion::UnaryApplyIf< Fusion::CastAndAttach<Base> , TargetOperation> > castAndAttach;
      };

      template <class Base, class Source>
      struct From<Base,Source,false>
      {
        template <class Arg>
        From(Arg&){}

        template <class... Targets>
        void to(Targets&...)
        {}
      };

      template <class Base, class Source>
      static auto from(Source& source) -> From<Base,Source>
      {
        return From<Base,Source>(source);
      }
    };

    template <class SourceOperation, class TargetOperation = SourceOperation>
    struct Deconnector
    {
      template <class Base, class Source, bool valid = isTrue< Apply<SourceOperation,Source> >() >
      struct From
      {
        From(Source& source)
          : castAndDetach(source)
        {}

        template <class... Targets>
        void to(Targets&... targets)
        {
          castAndDetach(targets...);
        }

        Fusion::VariadicFunctor< Fusion::UnaryApplyIf< Fusion::CastAndDetach<Base> , TargetOperation> > castAndDetach;
      };

      template <class Base, class Source>
      struct From<Base,Source,false>
      {
        template <class Arg>
        From(Arg&){}

        template <class... Targets>
        void to(Targets&...)
        {}
      };

      template <class Base, class Source>
      static auto from(Source& source) -> From<Base,Source>
      {
        return From<Base,Source>(source);
      }
    };
  }
}

#endif // DUNE_FGLUE_FUSION_CONNECT_HH
