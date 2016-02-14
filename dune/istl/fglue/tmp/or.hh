#ifndef DUNE_FGLUE_TMP_OR_HH
#define DUNE_FGLUE_TMP_OR_HH

#include <type_traits>
#include "true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Logical "or" for meta-functions that return std::true_type or std::false_type.
    struct Or
    {
      template <class First, class Second>
      struct apply
      {
        using type = typename std::conditional< isTrue<First>() || isTrue<Second>() , True , False >::type;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_OR_HH
