// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_COMBINEDFUNCTOR_HH
#define DUNE_AMG_COMBINEDFUNCTOR_HH

#include <dune/common/tuples.hh>
namespace Dune
{
  namespace Amg
  {

    template<std::size_t i>
    struct ApplyHelper
    {
      template<class TT, class T>
      static void apply(TT tuple, const T& t)
      {
        get<i-1>(tuple) (t);
        ApplyHelper<i-1>::apply(tuple, t);
      }
    };
    template<>
    struct ApplyHelper<0>
    {
      template<class TT, class T>
      static void apply(TT tuple, const T& t)
      {}
    };

    template<typename T1, typename T2>
    class CombinedFunctor : public Tuple<T1,T2>
    {
    public:
      CombinedFunctor(const T1& t1=T1(), const T2& t2=T2())
        : Tuple<T1,T2>(t1, t2)
      {}

      template<class T>
      void operator()(const T& t)
      {
        ApplyHelper<tuple_size<Tuple<T1,T2> >::value>::apply(*this, t);
      }
    };

    /*
       template<typename T1, typename T2 = Nil, typename T3 = Nil,
           typename T4 = Nil, typename T5 = Nil,typename T6 = Nil,
           typename T7 = Nil, typename T8 = Nil, typename T9 = Nil>
       class CombinedFunctor : public Tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>
       {
       public:
       CombinedFunctor(const T1& t1=T1(), const T2& t2=T2(), const T3& t3=T3(),
                      const T4& t4=T4(), const T5& t5=T5(), const T6& t6=T6(),
                      const T7& t7=T7(), const T8& t8=T8(), const T9& t9=T9())
        : Tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>(t1, t2, t3,
                                           t4, t5, t6,
                                           t7, t8, t9)
       {}

       template<class T>
       void operator()(const T& t)
       {
        ApplyHelper<tuple_size<Tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9> >::value>::apply(*this, t);
       }
       };

     */
  } //namespace Amg
} // namespace Dune
#endif
