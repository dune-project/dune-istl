// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <config.h>

#ifdef TEST_WITHOUT_BLAS
#undef HAVE_BLAS
#endif

#include <iostream>
#include <cmath>

#include <dune/common/simd/loop.hh>
#if HAVE_VC
#include <dune/common/simd/vc.hh>
#endif
#include <dune/common/alignedallocator.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/blockkrylov/matrixalgebra.hh>
#include <dune/istl/blockkrylov/utils.hh>
#include <dune/istl/blockkrylov/blockinnerproduct.hh>

using namespace Dune;

// checks whether all methods compile
template<class Algebra>
TestSuite testAlgebra(){
  TestSuite tsuite;
  using std::sqrt;
  std::cout << "Testing " << className<Algebra>() << std::endl;
  using X = typename Algebra::vector_type;
  using field_type = typename Algebra::field_type;
  using scalar_type = typename Algebra::scalar_type;
  SeqBlockInnerProduct<Algebra> bip;

  // test normalization
  X x(100);
  fillRandom(x, field_type(0.0) == 0.0);
  X y = x;
  Algebra rho = bip.bnormalize(x, x).get();
  Algebra prod = Algebra::dot(x,x);

  prod.scale(-1.0);
  prod.add(Algebra::identity());
  tsuite.check(prod.frobenius_norm() < sqrt(std::numeric_limits<scalar_type>::epsilon()), "Normalized vectoris not normalized");

  rho.axpy(-1.0, y, x);
  tsuite.check(Simd::allTrue(y.two_norm() < sqrt(std::numeric_limits<scalar_type>::epsilon())), "Normalization relation is not satisfied");

  // test orthogonalization (tests dot and axpy for nonsymmetric matrices)
  fillRandom(y, field_type(0.0) == 0.0);
  Algebra xy = Algebra::dot(y,x);
  xy.axpy(-1.0, y, x);
  xy = Algebra::dot(y,x);
  tsuite.check(xy.frobenius_norm() < sqrt(std::numeric_limits<scalar_type>::epsilon()), "dot product failed");

  return tsuite;
}

template<class field_type>
using AlignedBlockVector = Dune::BlockVector<field_type, Dune::AlignedAllocator<field_type>>;

int main(){
  TestSuite tsuite;
  Hybrid::forEach(std::tuple<LoopSIMD<double,8>,
                  LoopSIMD<double,8,32>
#if HAVE_VC
                  ,Vc::SimdArray<double,8>
                  ,LoopSIMD<Vc::Vector<double>, 8/Simd::lanes<Vc::Vector<double>>()>
#endif
                  >{},
                  [&](auto ft){
                    Hybrid::forEach(std::index_sequence<1,2,4,8>{},
                                    [&](auto p){
                                      using Algebra = ParallelMatrixAlgebra<AlignedBlockVector<decltype(ft)>, p>;
                                      tsuite.subTest(testAlgebra<Algebra>());
                                    });
                  });
  return 0;
}
