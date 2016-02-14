#ifndef DUNE_FGLUE_FUSION_VARIADIC_FUNCTOR_HH
#define DUNE_FGLUE_FUSION_VARIADIC_FUNCTOR_HH

#include <utility>

#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

namespace Dune
{
  namespace FGlue
  {
    namespace Fusion
    {
      namespace VariadicFunctorDetail
      {
        template <class,class...> struct Apply;

        template <class Functor, class Arg, class... Args>
        struct Apply<Functor,Arg,Args...>
        {
          static void apply(Functor& f, Arg& arg, Args&... args)
          {
            f(arg);
            Apply<Functor,Args...>::apply(f,args...);
          }
        };

        template <class Functor, class Arg>
        struct Apply<Functor,Arg>
        {
          static void apply(Functor& f, Arg& arg)
          {
            f(arg);
          }
        };
      }

      template <class Functor>
      struct VariadicFunctor
      {
        template <class... Args>
        VariadicFunctor(Args&&... args)
          : f_(std::forward<Args>(args)...)
        {}

        template <class... Args>
        void operator()(Args&&... args)
        {
          VariadicFunctorDetail::Apply<Functor,Args...>::apply(f_,args...);
        }

        Functor f_;
      };
    }
  }
}

#endif // DUNE_FGLUE_FUSION_VARIADIC_FUNCTOR_HH
