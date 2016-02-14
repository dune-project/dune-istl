#ifndef DUNE_FGLUE_TMP_STORE_IF_HH
#define DUNE_FGLUE_TMP_STORE_IF_HH

#include <type_traits>

#include <dune/common/typetraits.hh>

#include "apply.hh"
#include "is_base_of.hh"
#include "is_derived_from.hh"
#include "true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Stores Arg if Operation::template apply<Arg>::type evaluates to True, else stores Empty.
    template <class Operation>
    struct StoreIf
    {
      template <class Arg>
      struct apply
      {
        using type = typename std::conditional< isTrue< Apply<Operation,Arg> >() , Arg , Empty>::type;
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_STORE_IF_HH
