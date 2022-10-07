// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_COMBINEDFUNCTOR_HH
#define DUNE_AMG_COMBINEDFUNCTOR_HH

#include <tuple>

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
        std::get<i-1>(tuple) (t);
        ApplyHelper<i-1>::apply(tuple, t);
      }
    };
    template<>
    struct ApplyHelper<0>
    {
      template<class TT, class T>
      static void apply([[maybe_unused]] TT tuple, [[maybe_unused]] const T& t)
      {}
    };

    template<typename T>
    class CombinedFunctor :
      public T
    {
    public:
      CombinedFunctor(const T& tuple_)
        : T(tuple_)
      {}

      template<class T1>
      void operator()(const T1& t)
      {
        ApplyHelper<std::tuple_size<T>::value>::apply(*this, t);
      }
    };


  } //namespace Amg
} // namespace Dune
#endif
