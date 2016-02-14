#ifndef DUNE_FGLUE_TMP_APPLY_HH
#define DUNE_FGLUE_TMP_APPLY_HH

namespace Dune
{
  namespace FGlue
  {
    /// @cond
    namespace ApplyImpl
    {
      template <class...> struct Apply;

      template <class Operation>
      struct Apply<Operation>
      {
        using type = typename Operation::template apply<>::type;
      };

      template <class Operation, class... Args>
      struct Apply<Operation,Args...>
      {
        using type = typename Operation::template apply<Args...>::type;
      };
    }
    /// @endcond

    //! Convenient access to Operation::apply<Args...>::type.
    template <class... OperationAndArgs>
    using Apply = typename ApplyImpl::Apply<OperationAndArgs...>::type;
  }
}

#endif // DUNE_FGLUE_TMP_APPLY_HH
