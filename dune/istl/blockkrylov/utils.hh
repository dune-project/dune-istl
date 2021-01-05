// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_UTILS_HH
#define DUNE_ISTL_BLOCKKRYLOV_UTILS_HH

#include <dune/common/simd/interface.hh>

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

}

#endif
