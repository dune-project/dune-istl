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
#include <dune/common/parameterizedobject.hh>

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


namespace Dune {
  namespace {
    template<class Tag, std::size_t index>
    struct Registry;
  }

  namespace {
    template<template<class> class Base, class V, class Tag, typename... Args>
    auto registry_get(Tag , std::string name, Args... args)
    {
      constexpr auto count = getCounter(Tag);
      std::shared_ptr<Base<V> > result;
      Dune::Hybrid::forEach(std::make_index_sequence<count>{},
                            [&](auto index) {
                              using Reg = Registry<Tag, index>;
                              if(!result && Reg::name() == name) {
                                result = Reg::getCreator()(Dune::MetaType<V>{}, args...);
                              }
                            });
      return result;
    }

    /*
      Register all creators from the registry in the Parameterizedobjectfactory An
      object of V is passed in the creator ans should be used to determine the
      template arguments.
    */
    template<class V, class Type, class Tag, class... Args>
    int addRegistryToFactory(Dune::ParameterizedObjectFactory<Type(Args...), std::string>& factory,
                              Tag){
      constexpr auto count = getCounter(Tag);
      Dune::Hybrid::forEach(std::make_index_sequence<count>{},
                            [&](auto index) {
                              // we first get the generic lambda
                              // and later specialize it with given parameters.
                              // doing all at once lead to an ICE woth g++-6
                              using Reg = Registry<Tag, index>;
                              auto genericcreator = Reg::getCreator();
                              factory.define(Reg::name(), [genericcreator](Args... args){
                                                            return genericcreator(V{}, args...);
                                                          });
                            });
      return count;
    }
  } // end anonymous namespace
} // end namespace Dune

#endif // DUNE_ISTL_COMMON_REGISTRY_HH
