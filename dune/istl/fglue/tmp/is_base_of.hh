#ifndef DUNE_FGLUE_TMP_IS_BASE_OF_HH
#define DUNE_FGLUE_TMP_IS_BASE_OF_HH

#include <type_traits>

#include "apply.hh"
#include "chain.hh"
#include "not.hh"
#include "true_false.hh"
#include "variadic.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Meta-function that checks if its argument is a base class of Derived.
    template <class Derived>
    struct IsBaseOf
    {
      template <class Base>
      struct apply
      {
        using type = typename std::conditional<std::is_base_of<Base,Derived>::value,True,False>::type;
      };
    };

    //! Meta-function that checks if its argument is not a base class of Derived.
    template <class Derived>
    using IsNotBaseOf = Apply< Chain , Not , IsBaseOf<Derived> >;
  }
}

#endif // DUNE_FGLUE_TMP_IS_BASE_OF_HH
