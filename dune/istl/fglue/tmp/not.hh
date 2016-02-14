#ifndef DUNE_FGLUE_TMP_NOT_HH
#define DUNE_FGLUE_TMP_NOT_HH

#include <type_traits>
#include "true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Logical "not" for meta-functions.
    struct Not
    {
      template <class Arg>
      struct apply
      {
        using type = typename std::conditional< isTrue<Arg>() , False , True >::type;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_NOT_HH
