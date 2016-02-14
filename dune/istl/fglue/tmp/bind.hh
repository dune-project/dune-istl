#ifndef DUNE_FGLUE_TMP_BIND_HH
#define DUNE_FGLUE_TMP_BIND_HH

#include "apply.hh"

namespace Dune
{
  namespace FGlue
  {
    /// @cond
    namespace Impl{ template <class...> struct Bind; }
    /// @endcond

    //! Bind Args... to Operation::apply. Thus apply becomes a nullary meta-function.
    struct Bind
    {
      template <class...> struct apply;

      template <class Operation>
      struct apply<Operation>
      {
        using type = Impl::Bind<Operation>;
      };

      template <class Operation, class... Args>
      struct apply<Operation,Args...>
      {
        using type = Impl::Bind<Operation,Args...>;
      };
    };

    /// @cond
    namespace Impl
    {
      template <class... Args> struct Bind;

      template <class Operation>
      struct Bind<Operation>
      {
        template <class...>
        struct apply
        {
          using type = Apply<Operation>;
        };
      };

      template <class Operation, class... Args>
      struct Bind<Operation,Args...>
      {
        template <class...>
        struct apply
        {
          using type = Apply<Operation,Args...>;
        };
      };
    }
    /// @endcond
  }
}

#endif // DUNE_FGLUE_TMP_BIND_HH
