#ifndef DUNE_FGLUE_TMP_MAKE_HH
#define DUNE_FGLUE_TMP_MAKE_HH

#include "apply.hh"

namespace Dune
{
  namespace FGlue
  {
    template <template <class...> class Operation>
    struct Make
    {
      template <class... Args>
      struct apply
      {
        using type = Operation<Args...>;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_MAKE_HH
