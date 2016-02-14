#ifndef DUNE_FGLUE_TMP_DELAY_HH
#define DUNE_FGLUE_TMP_DELAY_HH

#include "apply.hh"

namespace Dune
{
  namespace FGlue
  {
    template <class Operation>
    struct Delay
    {
      template <class... Operations>
      struct apply;

      template <class FirstOp>
      struct apply<FirstOp>
      {
        struct type
        {
          template <class... Args>
          struct apply
          {
            using type = Apply<Operation,Apply<FirstOp,Args...> >;
          };
        };
      };

      template <class FirstOp,class SecondOp>
      struct apply<FirstOp,SecondOp>
      {
        struct type
        {
          template <class... Args>
          struct apply
          {
            using type = Apply<Operation,Apply<FirstOp,Args...>,Apply<SecondOp,Args...> >;
          };
        };
      };
    };
  }
}

#endif // DUNE_FGLUE_TMP_DELAY_HH
