#ifndef DUNE_FGLUE_FUSION_APPLY_IF_HH
#define DUNE_FGLUE_FUSION_APPLY_IF_HH

#include <utility>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    namespace Fusion
    {
      template <class Operation, class CompileTimeDecider>
      struct UnaryApplyIf
      {
        template <class... Args>
        UnaryApplyIf(Args&&... args)
          : f_(std::forward<Args>(args)...)
        {}

        template <class Arg,
                  typename std::enable_if< !isTrue< Apply< CompileTimeDecider , typename std::decay<Arg>::type > >() >::type* = nullptr >
        void operator()(Arg&& arg)
        {}

        template <class Arg,
                  typename std::enable_if< isTrue< Apply< CompileTimeDecider , typename std::decay<Arg>::type > >() >::type* = nullptr >
        auto operator()(Arg&& arg) -> decltype( std::declval<Operation>()(arg) )
        {
          f_(arg);
        }

      private:
        Operation f_;
      };

    }
  }
}

#endif // DUNE_FGLUE_FUSION_APPLY_IF_HH
