#ifndef DUNE_FGLUE_TMP_CREATE_MISSING_BASE_CLASSES_HH
#define DUNE_FGLUE_TMP_CREATE_MISSING_BASE_CLASSES_HH

#include "and.hh"
#include "apply.hh"
#include "delay.hh"
#include "is_base_of.hh"
#include "make.hh"
#include "or.hh"
#include "store_if.hh"
#include "variadic.hh"

namespace Dune
{
  namespace FGlue
  {
    template <class... Derived>
    using IsBaseOfOneOf = Apply< Variadic< Make<IsBaseOf> , Delay<Or> > , Derived... >;

    template <class... Derived>
    using IsNotBaseOfAnyOf = Apply< Make<IsNotBaseOf> , Delay<And> , Derived... >;


    template <class Operation, class... Derived>
    using EnableBaseClassesIf = Apply< Variadic<StoreIf<Operation>,Compose> , Derived...>;


    template <class... Derived>
    struct BaseOf
    {
      template <class... OtherBases>
      using NotBaseOf = StoreIf< Apply< Delay<And> , IsBaseOfOneOf<Derived...> , IsNotBaseOfAnyOf<OtherBases...> > >;
    };

    template <class... BaseClassCandidates>
    struct CreateMissingBases
    {
      template <class... Derived>
      struct BaseOf
      {
        template <class... OtherBases>
        using NotBaseOf = Apply< Variadic< typename FGlue::BaseOf<Derived...>::template NotBaseOf<OtherBases...> , Compose > , BaseClassCandidates... >;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_CREATE_MISSING_BASE_CLASSES_HH
