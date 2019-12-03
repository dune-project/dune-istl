#include <cassert>

#include <dune/common/classname.hh>
#include <dune/istl/common/registry.hh>

template<template<class> class Thing>
auto default_thing_creator()
{
  return [] (auto m, int i) {
    return std::make_shared<Thing<typename decltype(m)::type> >(i);
  };
}

struct ThingTag {};

template<class V>
struct ThingBase {
  virtual std::string do_something() = 0;
};

//////////////////////////////////////////////////////////////////////
//
// thing A
//

template<class V> struct ThingA : ThingBase<V> {
  int _v;
  ThingA(int v) : _v(v) {}
  virtual std::string do_something() {
    return Dune::className<V>() + " A(" + std::to_string(_v) + ")";
  }
};
registry_put(ThingTag, "A", default_thing_creator<ThingA>());

//////////////////////////////////////////////////////////////////////
//
// thing B
//

template<class V> struct ThingB : ThingBase<V> {
  int _v;
  std::string arg;
  ThingB(int v, std::string arg_) : _v(v), arg(arg_) {}
  virtual std::string do_something() {
    return Dune::className<V>() + " B(" + arg + "," + std::to_string(_v) + ")";
  }
};
registry_put(ThingTag, "B", [] (auto m, int i) {
    return std::make_shared<ThingB<typename decltype(m)::type> >(i,"dynamic");
  });

template<typename T>
bool check(std::string key, int v, std::string reference)
{
  std::shared_ptr<ThingBase<T>> p = registry_get<ThingBase, T>(ThingTag{}, key, v);
  std::string s = p->do_something();
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
  success &= (registry_get<ThingBase, int>(ThingTag{}, "C", 3) == nullptr);
  success &= check<double>("A", 4, "d A(4)");

  std::cout << "SUCCESS: " << success << std::endl;


  //try to build a parameterizedobjectfactory from registry
  Dune::ParameterizedObjectFactory<std::shared_ptr<ThingBase<int>>(int)> fac;
  addRegistryToFactory<int,Dune::MetaType<int>>(fac, ThingTag{});
  std::cout << fac.create("A", 1)->do_something() << std::endl;
  return success ? 0 : 1;
}
