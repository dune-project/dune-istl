// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <dune/common/memory/blocked_allocator.hh>
#include <dune/istl/vector/host.hh>

#include <tbb/tbb.h>

template<typename F, typename A=std::allocator<F> >
int testVector()
{

  typedef Dune::ISTL::Vector<F,A> V;

  V v(2000000);

  //v.setChunkSize(40000);

  std::cout << v.chunkSize() << std::endl;

  //V v2;
  //v2.setSize(2000);

  //V v3(std::move(v2));

  //v3 += v;

  std::iota(v.begin(),v.end(),0);

  v -= 3.0;
  v += 3.0;

  V v2(2000000);
  std::fill(v2.begin(),v2.end(),1.1);

  v += v2;
  v *= v2;

  v.axpy(17.3,v2);

  //v3 = v;

  for (int i = 0; i < 20; ++i)
    {
      std::cout << v.one_norm() << std::endl;
      std::cout << v.two_norm() << std::endl;
      std::cout << v.infinity_norm() << std::endl;
      v += v;
    }

  v /= 2.0;
  v *= 0.5;

  std::cout << v.dot(v2) << std::endl;


  //v.axpy(-1.0,v3);

  return 0;
}


int main(int argc, char** argv)
{
  tbb::task_scheduler_init(argc > 1 ? atoi(argv[1]) : tbb::task_scheduler_init::automatic);
  return testVector<double,Dune::Memory::blocked_cache_aligned_allocator<double,std::size_t,16> >();
}
