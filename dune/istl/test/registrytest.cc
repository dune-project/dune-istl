// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <cassert>

#ifndef DISABLE_CXA_DEMANGLE
#define DISABLE_CXA_DEMANGLE 1
#endif

#include <dune/common/classname.hh>
#include <dune/istl/common/registry.hh>

template<template<class> class Thing>
auto defaultThingCreator()
{
  return [] (auto m, int i) {
    return std::make_shared<Thing<typename decltype(m)::type> >(i);
  };
}

struct ThingTag {};

template<class V>
struct ThingBase {
  virtual std::string doSomething() = 0;
  virtual ~ThingBase() {}
};

//////////////////////////////////////////////////////////////////////
//
// thing A
//

template<class V> struct ThingA : ThingBase<V> {
  int _v;
  ThingA(int v) : _v(v) {}
  virtual std::string doSomething() {
    return Dune::className<V>() + " A(" + std::to_string(_v) + ")";
  }
};
namespace Dune {
  DUNE_REGISTRY_PUT(ThingTag, "A", defaultThingCreator<ThingA>());
}

//////////////////////////////////////////////////////////////////////
//
// thing B
//

template<class V> struct ThingB : ThingBase<V> {
  int _v;
  std::string arg;
  ThingB(int v, std::string arg_) : _v(v), arg(arg_) {}
  virtual std::string doSomething() {
    return Dune::className<V>() + " B(" + arg + "," + std::to_string(_v) + ")";
  }
};
namespace Dune {
DUNE_REGISTRY_PUT(ThingTag, "B", [] (auto m, int i) {
    return std::make_shared<ThingB<typename decltype(m)::type> >(i,"dynamic");
  });
}

template<typename T>
bool check(std::string key, int v, std::string reference)
{
  std::shared_ptr<ThingBase<T>> p = Dune::registryGet<ThingBase, T>(ThingTag{}, key, v);
  std::string s = p->doSomething();
  bool r = (s == reference);
  std::cout << "CHECK " << s
            << " / " << reference
            << " -> " << r << std::endl;
  return r;
}

template<typename F>
bool checkDynamic(F& factory, std::string key, int v, std::string reference)
{
  auto p = factory.create(key,v);
  std::string s = p->doSomething();
  bool r = (s == reference);
  std::cout << "CHECK " << s
            << " / " << reference
            << " -> " << r << std::endl;
  return r;
}

//////////////////////////////////////////////////////////////////////
int main() {
  bool success = true;

  success &= check<int>("A", 1, "i A(1)");
  success &= check<int>("B", 2, "i B(dynamic,2)");
  success &= (Dune::registryGet<ThingBase, int>(ThingTag{}, "C", 3) == nullptr);
  success &= check<double>("A", 4, "d A(4)");

  //try to build a parameterizedobjectfactory from registry
  Dune::ParameterizedObjectFactory<std::shared_ptr<ThingBase<int>>(int)> fac;
  Dune::addRegistryToFactory<Dune::MetaType<int>>(fac, ThingTag{});
  success &= checkDynamic(fac, "A", 1, "i A(1)");
  success &= checkDynamic(fac, "B", 2, "i B(dynamic,2)");

  std::cout << "SUCCESS: " << success << std::endl;

  return success ? 0 : 1;
}
