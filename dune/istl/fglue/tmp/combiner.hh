#ifndef DUNE_FGLUE_TMP_COMBINER_HH
#define DUNE_FGLUE_TMP_COMBINER_HH

#include <type_traits>

#include "dune/common/typetraits.hh"

#include "apply.hh"

namespace Dune
{
  namespace FGlue
  {
    //! Always returns Empty.
    struct DefaultCombiner
    {
      template <class First, class Second>
      struct apply
      {
        using type = Empty;
      };
    };


    /*!
      @brief Composition of return types from meta-functions.

      If both meta-functions return types different from Empty than a class that inherits from
      both results is returned.
      If both return Empty then returns Empty
      Else returns the result that differs from Empty
     */
    struct Compose
    {
      template <class First, class Second,
                bool onlyFirst = std::is_base_of< Empty , First >::value ||
                ( std::is_base_of< First , Second >::value && !std::is_same< First , Second >::value ),
                bool onlySecond = std::is_base_of< Empty , Second >::value || std::is_base_of< Second , First >::value>
      struct apply
      {
        using type = Empty;
      };

      template <class First, class Second>
      struct apply<First,Second,false,true>
      {
        using type = First;
      };

      template <class First, class Second>
      struct apply<First,Second,true,false>
      {
        using type = Second;
      };

      template <class First, class Second>
      struct apply<First,Second,false,false>
      {
        struct type :
            First,
            Second
        {};
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_COMBINER_HH
