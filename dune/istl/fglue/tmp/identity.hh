#ifndef DUNE_FGLUE_TMP_IDENTITY_HH
#define DUNE_FGLUE_TMP_IDENTITY_HH

namespace Dune
{
  namespace FGlue
  {
    struct Identity
    {
      template <class Arg>
      struct apply
      {
        using type = Arg;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_IDENTITY_HH
