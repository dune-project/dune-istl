#ifndef DUNE_OPTIONAL_HH
#define DUNE_OPTIONAL_HH

#include <functional>
#include <string>
#include <utility>

#include <dune/common/typetraits.hh>

namespace Dune
{
  //! @cond
  class InverseOperatorResult;

  namespace Try
  {
    namespace MemFn
    {
      template <class Type>
      using terminate = decltype( std::declval<Type>().terminate() );

      template <class Connector, class ToConnect>
      using connect = decltype(std::declval<Connector>().connect(std::declval<ToConnect>()));

      template <class Type>
      using minimalDecreaseAchieved = decltype( std::declval<Type>().minimalDecreaseAchieved() );

      template <class Type>
      using restart = decltype(std::declval<Type>().restart());

      template <class Type>
      using reset = decltype( std::declval<Type>().reset() );

      template <class Type, class... Args>
      using reset_NArgs = decltype( std::declval<Type>().reset( std::declval<Args>()... ) );

      template <class Type, class... Args>
      using init = decltype( std::declval<Type>().init( std::declval<Args>()... ) );

      template <class Type, class Arg>
      using postProcess = decltype( std::declval<Type>().postProcess( std::declval<Arg>() ) );

      template <class Type>
      using writeToInverseOperatorResult = decltype( std::declval<Type>().write( std::declval<InverseOperatorResult&>() ) );

      template <class Type, class Arg>
      using setRelativeAccuracy = decltype( std::declval<Type>().setRelativeAccuracy( std::declval<Arg>() ) );

      template <class Type>
      using name = decltype( std::declval<Type>().name() );

      template <class Type>
      using errorEstimate = decltype( std::declval<Type>().errorEstimate() );
    }
  }
  //! @endcond

  /*!
    Functions in this namespace use SFINAE to check if functions, that are to be executed, actually exist.
    If this is not the case fallback implementations are provided.
   */
  namespace Optional
  {
    //! @cond
    namespace Detail
    {
      template <class Connector, class ToConnect, class = void>
      struct Connect
      {
        static void apply(Connector&, const ToConnect&)
        {}
      };

      template <class Connector, class ToConnect>
      struct Connect< Connector, ToConnect, void_t<Try::MemFn::connect<Connector,ToConnect> > >
      {
        static void apply(Connector& connector, const ToConnect& toConnect)
        {
          connector.connect(toConnect);
        }
      };


      template <class Type, class = void>
      struct Restart
      {
        static bool apply(const Type&)
        {
          return false;
        }
      };

      template <class Type>
      struct Restart< Type , void_t< Try::MemFn::restart<Type> > >
      {
        static bool apply(const Type& t)
        {
          return t.restart();
        }
      };


      template <class Type, class = void>
      struct Terminate
      {
        static bool apply(const Type&)
        {
          return false;
        }
      };

      template <class Type>
      struct Terminate< Type , void_t< Try::MemFn::terminate<Type> > >
      {
        static bool apply(const Type& t)
        {
          return t.terminate();
        }
      };


      template <class Type, class = void>
      struct Reset
      {
        static void apply(const Type&)
        {
        }
      };

      template <class Type>
      struct Reset< Type , void_t< Try::MemFn::reset<Type> > >
      {
        static void apply(Type& t)
        {
          t.reset();
        }
      };


      template <class Type, class Arg1, class Arg2, class = void>
      struct Reset_2Args
      {
        static void apply(const Type&, const Arg1&, const Arg2&)
        {
        }
      };

      template <class Type, class Arg1, class Arg2>
      struct Reset_2Args< Type, Arg1, Arg2, void_t< Try::MemFn::reset_NArgs<Type,Arg1&,Arg2&> > >
      {
        static void apply(Type& t, Arg1& x, Arg2& b)
        {
          t.reset(x,b);
        }
      };



      template <class Type, class Arg1, class Arg2, class = void>
      struct Init_2Args
      {
        static void apply(const Type&, const Arg1&, const Arg2&)
        {
        }
      };

      template <class Type, class Arg1, class Arg2>
      struct Init_2Args< Type, Arg1, Arg2, void_t< Try::MemFn::init<Type,Arg1&,Arg2&> > >
      {
        static void apply(Type& t, Arg1& x, Arg2& b)
        {
          t.init(x,b);
        }
      };


      template <class Type, class Arg1, class = void>
      struct PostProcess
      {
        static void apply(const Type&, const Arg1&)
        {
        }
      };

      template <class Type, class Arg1>
      struct PostProcess< Type, Arg1, void_t< Try::MemFn::postProcess<Type,Arg1&> > >
      {
        static void apply(Type& t, Arg1& x)
        {
          t.postProcess(x);
        }
      };


      template <class Type, class = void>
      struct WriteToInverseOperatorResult
      {
        static void apply(const Type&, const InverseOperatorResult&)
        {
        }
      };

      template <class Type>
      struct WriteToInverseOperatorResult< Type, void_t< Try::MemFn::writeToInverseOperatorResult<Type> > >
      {
        static void apply(Type& t, InverseOperatorResult& result)
        {
          t.write(result);
        }
      };


      template <class Type, class Arg, class = void>
      struct SetRelativeAccuracy
      {
        static void apply(const Type&, const Arg&)
        {
        }
      };

      template <class Type, class Arg>
      struct SetRelativeAccuracy< Type, Arg, void_t< Try::MemFn::setRelativeAccuracy<Type,Arg> > >
      {
        static void apply(Type& t, Arg accuracy)
        {
          t.setRelativeAccuracy(accuracy);
        }
      };


      template <class Type, class = void>
      struct Name
      {
        static std::string apply(const Type&)
        {
          return "unnamed";
        }
      };

      template <class Type>
      struct Name< Type, void_t< Try::MemFn::name<Type> > >
      {
        static std::string apply(const Type& t)
        {
          return t.name();
        }
      };


      template <class Type, class = void>
      struct ErrorEstimate
      {
        static double apply(const Type&)
        {
          return 1;
        }
      };

      template <class Type>
      struct ErrorEstimate< Type, void_t< Try::MemFn::errorEstimate<Type> > >
      {
        static auto apply(const Type& t) -> Try::MemFn::errorEstimate<Type>
        {
          return t.errorEstimate();
        }
      };
    }
    //! @endcond


    /// Returns the result of `%t.errorEstimate()` if present, else returns 1.
    template <class Type>
    auto errorEstimate(const Type& t) -> decltype( Detail::ErrorEstimate<Type>::apply(std::declval<Type>()) )
    {
      return Detail::ErrorEstimate<Type>::apply(t);
    }


    /// Call `%t.name()` if present, else returns '"unnamed"'.
    template <class Type>
    std::string name(const Type& t)
    {
      return Detail::Name<Type>::apply(t);
    }


    /// Call `t.setRelativeAccuracy(accuracy)` if present.
    template <class Type, class Arg>
    void setRelativeAccuracy(Type& t, Arg accuracy)
    {
      Detail::SetRelativeAccuracy<Type,Arg>::apply(t,accuracy);
    }


    /// Call `t.write(result)` if present.
    template <class Type>
    void write(Type& t, InverseOperatorResult& result)
    {
      Detail::WriteToInverseOperatorResult<Type>::apply(t,result);
    }


    /// Call `t.postProcess(x)` if present.
    template <class Type, class Arg>
    void postProcess(Type& t, Arg& x)
    {
      Detail::PostProcess<Type,Arg>::apply(t,x);
    }


    /// Call `t.init(x,b)` if present.
    template <class Type, class Arg1, class Arg2>
    void init(Type& t, Arg1& x, Arg2& b)
    {
      Detail::Init_2Args<Type,Arg1,Arg2>::apply(t,x,b);
    }


    /// Call `%t.reset()` if present.
    template <class Type>
    void reset(Type& t)
    {
      Detail::Reset<Type>::apply(t);
    }


    /// Call `t.reset(x,b)` if present.
    template <class Type, class Arg1, class Arg2>
    void reset(Type& t, Arg1& x, Arg2& b)
    {
      Detail::Reset_2Args<Type,Arg1,Arg2>::apply(t,x,b);
    }


    /// Returns the result of `%t.terminate()` if present, else returns false.
    template <class Type>
    bool terminate(const Type& t)
    {
      return Detail::Terminate<Type>::apply(t);
    }


    /// Returns the result of `%t.restart()` if present, else return false.
    template <class Type>
    bool restart(const Type& t)
    {
      return Detail::Restart<Type>::apply(t);
    }


    /// Call `connector.connect(toConnect)` if present.
    template <class Connector, class ToConnect>
    void connect(Connector& connector, const ToConnect& toConnect)
    {
      Detail::Connect<Connector,ToConnect>::apply(connector,toConnect);
    }
  }
}

#endif // DUNE_OPTIONAL_HH
