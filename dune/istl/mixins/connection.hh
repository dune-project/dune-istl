#ifndef DUNE_MIXINS_CONNECTION_HH
#define DUNE_MIXINS_CONNECTION_HH

#include <functional>
#include <type_traits>
#include <vector>

namespace Dune
{
  /** @addtogroup MixinGroup
   *  @{
   */
  namespace Mixin
  {
    /// Helper class to connect mixins and compatible other classes.
    template <class Arg, class Return=void>
    struct Connection
    {
      /// Attached function.
      void attach( std::function<Return(Arg)> f)
      {
        connected_.push_back( std::move(f) );
      }

      /// Call `f(arg)` for all attached functions `f`.
      void update(Arg arg)
      {
        for( auto& f : connected_)
          f(arg);
      }

      /// Remove all attached functions.
      void clear()
      {
        connected_.clear();
      }

    private:
      std::vector< std::function<Return(Arg)> > connected_ = {};
    };
  }

  namespace Optional
  {
    namespace Mixin
    {
      /// Call `static_cast<Mixin&>(callee).attach(toAttach)`, if `Callee` and `ToAttach` are derived from `Mixin`.
      template <class Callee, class ToAttach, class Mixin,
                bool valid = std::is_base_of<Mixin,Callee>::value && std::is_base_of<Mixin,ToAttach>::value>
      struct SingleAttach
      {
        static void apply(Callee& callee, ToAttach& toAttach)
        {
          static_cast<Mixin&>(callee).attach( toAttach );
        }
      };

      /// Specialization for the case that `Callee` is not derived from `Mixin`. Does nothing.
      template <class Callee,class ToAttach, class Mixin>
      struct SingleAttach<Callee,ToAttach,Mixin,false>
      {
        static void apply(Callee&, ToAttach&)
        {}
      };

      /// Attach an arbitrary number of mixin classes.
      template <class... Mixins>
      struct Attach;

      /// @cond
      template <class Mixin, class... Mixins>
      struct Attach<Mixin,Mixins...>
      {
        template <class Callee, class ToAttach, class... Other>
        static void apply(Callee& callee, ToAttach& toAttach, Other&... other)
        {
          apply(callee,toAttach);
          apply(callee,other...);
        }

        template <class Callee>
        static void apply(Callee&)
        {}

        template <class Callee, class ToAttach>
        static void apply(Callee& callee, ToAttach& toAttach)
        {
          SingleAttach<Callee,ToAttach,Mixin>::apply(callee,toAttach);
          Attach<Mixins...>::apply(callee,toAttach);
        }
      };

      template <>
      struct Attach<>
      {
        template <class Callee, class ToAttach>
        static void apply(Callee&, ToAttach&)
        {}
      };
      /// @endcond

      /// Calls `SingleAttach<Callee,ToAttach,Mixin>::apply(callee,toAttach)`.
      template <class Mixin, class Callee, class ToAttach>
      void attach(Callee& callee, ToAttach& toAttach)
      {
        SingleAttach<Callee,ToAttach,Mixin>::apply(callee,toAttach);
      }
    }
  }
  /** @} */
}

#endif // DUNE_MIXINS_CONNECTION_HH
