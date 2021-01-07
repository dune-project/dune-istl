// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_UTILS_HH
#define DUNE_ISTL_BLOCKKRYLOV_UTILS_HH

/** \file
 * \brief Provides utility functions for block Krylov methods
 */

#include <dune/common/hybridutilities.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/simd/interface.hh>

#include <dune/istl/solver.hh>

namespace Dune {

  template<class V>
  std::enable_if_t<IsNumber<V>::value>
  fillRandom(V& x, Simd::Mask<V> mask){
    using scalar = Simd::Scalar<V>;
    for(size_t l=0;l<Simd::lanes(x); ++l){
      if(Simd::lane(l,mask))
        Simd::lane(l,x) = scalar(std::rand())/scalar(RAND_MAX);
    }
  }

  template<class V>
  std::enable_if_t<!IsNumber<V>::value>
  fillRandom(V& x, Simd::Mask<typename V::field_type> mask){
    for(auto& r : x){
      fillRandom(r, mask);
    }
  }

  template<class S>
  size_t countTrue(const S& x){
    size_t sum = 0;
    for(size_t l=0;l<Simd::lanes(x); ++l){
      if(Simd::lane(l,x))
        sum++;
    }
    return sum;
  }

  namespace {
    template<size_t S, size_t... P>
    constexpr auto dividersOfImpl(std::index_sequence<P...>){
      auto result = std::tuple_cat(std::conditional_t<(S % (P+1) == 0), std::tuple<std::integral_constant<size_t, P+1>>, std::tuple<>>{}...);
      return result;
    }

    template<class I, I... P>
    auto tuple_to_integer_sequence(std::tuple<std::integral_constant<I, P>...>){
      return std::integer_sequence<I, P...>{};
    }
  }

  template<size_t S>
  constexpr auto dividersOf(){
    return tuple_to_integer_sequence(dividersOfImpl<S>(std::make_index_sequence<S>()));
  }

  // solverfactory function to create block Krylov solvers
  template<template<class, size_t> class Solver>
  auto blockKrylovSolverCreator(){
    return [](auto typeList,
              const auto& linearOperator,
              const auto& scalarProduct,
              const auto& preconditioner,
              const ParameterTree& config)
    {
      using Domain = typename Dune::TypeListElement<0, decltype(typeList)>::type;
      using Range = typename Dune::TypeListElement<1, decltype(typeList)>::type;
      constexpr size_t K = Simd::lanes<typename Domain::field_type>();
      std::shared_ptr<InverseOperator<Domain, Range>> solver;
      Hybrid::switchCases(dividersOf<K>(),
                          config.get<size_t>("p", K),
                          [&](auto pp){
                            solver = std::make_shared<Solver<Domain, (pp.value)>>(linearOperator, scalarProduct, preconditioner, config);
                          },
                          [](auto...){
                             DUNE_THROW(Exception, "Invalid parameter P: P must be a divider of the SIMD width");
                          });
      return solver;
    };
  }

  template<class X, class Y>
  std::shared_ptr<X> dynamic_cast_or_throw(std::shared_ptr<Y> y){
    std::shared_ptr<X> x = std::dynamic_pointer_cast<X>(y);
    if(!x)
      throw std::bad_cast{};
    return x;
  }

}

#endif
