#ifndef DUNE_FGLUE_TRUE_FALSE_HH
#define DUNE_FGLUE_TRUE_FALSE_HH

#include <type_traits>

namespace Dune
{
  namespace FGlue
  {
    //! Template meta-function that always evaluates to std::true_type.
    using True = std::true_type;

    //! Template meta-function that always evaluates to std::false_type.
    using False = std::false_type;

    //! @return std::is_same<typename std::decay<Type>::type,True>::value;
    template <class Type>
    constexpr bool isTrue()
    {
      return std::is_same<typename std::decay<Type>::type,True>::value;
    }
  }
}

#endif // DUNE_FGLUE_TRUE_FALSE_HH
