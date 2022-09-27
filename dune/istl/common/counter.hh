// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_ISTL_COMMON_COUNTER_HH
#define DUNE_ISTL_COMMON_COUNTER_HH

#include <cassert>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>

#include <dune/common/typeutilities.hh>

constexpr std::size_t maxcount = 100;

#define DUNE_GET_COUNTER(Tag)                                                 \
  (counterFunc(Dune::PriorityTag<maxcount>{}, Tag{}, Dune::CounterImpl::ADLTag{}))

#define DUNE_INC_COUNTER(Tag)                                           \
  namespace {                                                           \
    namespace CounterImpl {                                               \
      constexpr std::size_t                                             \
      counterFunc(Dune::PriorityTag<DUNE_GET_COUNTER(Tag)+1> p, Tag, ADLTag)        \
      {                                                                 \
        return p.value;                                                 \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  static_assert(true, "unfudge indentation")

namespace Dune {
  namespace {

    namespace CounterImpl {

      struct ADLTag {};

      template<class Tag>
      constexpr std::size_t counterFunc(Dune::PriorityTag<0>, Tag, ADLTag)
      {
        return 0;
      }

    } // end namespace CounterImpl
  } // end empty namespace
} // end namespace Dune
#endif // DUNE_ISTL_COMMON_COUNTER_HH
