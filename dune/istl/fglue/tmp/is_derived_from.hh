#ifndef DUNE_FGLUE_TMP_IS_DERIVED_FROM_HH
#define DUNE_FGLUE_TMP_IS_DERIVED_FROM_HH

#include <type_traits>

#include "apply.hh"
#include "chain.hh"
#include "not.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Meta-function that checks if its argument is derived from Base.
    template <class Base>
    struct IsDerivedFrom
    {
      template <class Derived>
      struct apply
      {
        using type = typename std::conditional<std::is_base_of<Base,Derived>::value,True,False>::type;
      };
    };

    //! Meta-function that checks if its argument is not derived from Base
    template <class Base>
    using IsNotDerivedFrom = Apply< Chain , Not , IsDerivedFrom<Base> >;
  }
}

#endif // DUNE_FGLUE_TMP_IS_DERIVED_FROM_HH
