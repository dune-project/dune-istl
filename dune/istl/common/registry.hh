#ifndef DUNE_ISTL_COMMON_REGISTRY_HH
#define DUNE_ISTL_COMMON_REGISTRY_HH

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "counter.hh"

#include <dune/common/typelist.hh>
#include <dune/common/hybridutilities.hh>

namespace {
  template<class Tag, std::size_t index>
  struct Registry;
}

#define registry_put(Tag, id, ...)              \
  namespace {                                   \
    template<>                                  \
    struct Registry<Tag, getCounter(Tag)>       \
    {                                           \
      static auto getCreator()                  \
      {                                         \
        return __VA_ARGS__;                     \
      }                                         \
      static std::string name() { return id; }  \
    };                                          \
  }                                             \
  incCounter(Tag)

namespace {
  template<template<class> class Base, class V, class Tag, typename... Args>
  auto registry_get(Tag t, std::string name, Args... args)
  {
    constexpr auto count = getCounter(Tag);
    std::shared_ptr<Base<V> > result;
    Dune::Hybrid::forEach(std::make_index_sequence<count>{}, [&](auto index) {
        using Reg = Registry<Tag, index>;
        if(!result && Reg::name() == name) {
          result = Reg::getCreator()(Dune::MetaType<V>{}, args...);
        }
      });
    return result;
  }
}

#endif // DUNE_ISTL_COMMON_REGISTRY_HH
