#ifndef DUNE_FGLUE_TMP_CHAIN_HH
#define DUNE_FGLUE_TMP_CHAIN_HH

#include "apply.hh"

namespace Dune
{
  namespace FGlue
  {
    struct Chain
    {
      template <class F, class G>
      struct apply
      {
        struct type
        {
          template <class... Args>
          struct apply
          {
            using type = Apply< F , Apply<G,Args...> >;
          };
        };
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_CHAIN_HH
