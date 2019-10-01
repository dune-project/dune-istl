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

namespace {

  namespace Overloads {

    struct ADLTag {};

    template<class Tag>
    constexpr std::size_t counterFunc(Dune::PriorityTag<0>, Tag, ADLTag)
    {
      return 0;
    }

  } // end namespace Overloads
} // end empty namespace

#define getCounter(Tag)                                                 \
  (counterFunc(Dune::PriorityTag<maxcount>{}, Tag{}, ::Overloads::ADLTag{}))

#define incCounter(Tag)                                                 \
  namespace {                                                           \
    namespace Overloads {                                               \
      constexpr std::size_t                                             \
      counterFunc(Dune::PriorityTag<getCounter(Tag)+1> p, Tag, ADLTag)        \
      {                                                                 \
        return p.value;                                                 \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  static_assert(true, "unfudge indentation")

#endif // DUNE_ISTL_COMMON_COUNTER_HH
