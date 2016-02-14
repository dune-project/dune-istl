#ifndef DUNE_FGLUE_TMP_CONSTANT_HH
#define DUNE_FGLUE_TMP_CONSTANT_HH

namespace Dune
{
  namespace FGlue
  {
    template <class Operation>
    struct Constant
    {
      template <class...Args>
      struct apply
      {
        using type = Operation;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_CONSTANT_HH
