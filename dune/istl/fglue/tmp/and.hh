#ifndef DUNE_GLUE_TMP_AND_HH
#define DUNE_GLUE_TMP_AND_HH

#include <type_traits>
#include "true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    /// Logical "and" for meta-functions. Return True or False.
    class And
    {
    public:
      template <class First, class Second>
      struct apply
      {
         using type = typename std::conditional< isTrue<First>() && isTrue<Second>() , std::true_type , std::false_type >::type;
      };
    };
  }
}

#endif // DUNE_GLUE_TMP_AND_HH
