#ifndef DUNE_FGLUE_TMP_VARIADIC_APPLY_HH
#define DUNE_FGLUE_TMP_VARIADIC_APPLY_HH

#include "apply.hh"
#include "combiner.hh"
#include "identity.hh"

namespace Dune
{
  namespace FGlue
  {
    /*!
      @tparam Operation unary operation
      @tparam Combiner combines the individual results
     */
    template <class Operation, class Combiner = DefaultCombiner>
    struct Variadic
    {
      template <class... Args>
      struct apply;

      template<class Arg, class... Args>
      struct apply<Arg,Args...>
      {
        using type = Apply< Combiner , Apply<Operation,Arg>, Apply<Variadic<Operation,Combiner>,Args...> >;
      };

      template <class Arg>
      struct apply<Arg>
      {
        using type = Apply<Operation,Arg>;
      };
    };


    template <class Operation>
    using Binary2Variadic = Variadic< Identity , Operation >;
  }
}

#endif // DUNE_FGLUE_TMP_VARIADIC_APPLY_HH
