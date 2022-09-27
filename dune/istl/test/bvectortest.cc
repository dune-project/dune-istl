// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

// hack to ensure assert() does something
// really, assert() should not be used in unit tests.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <dune/common/classname.hh>
#if HAVE_MPROTECT
#include <dune/common/debugallocator.hh>
#endif
#include <dune/common/fvector.hh>
#include <dune/common/poolallocator.hh>
#include <dune/common/scalarvectorview.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/test/vectortest.hh>

template<typename VectorBlock, typename V>
void assign(VectorBlock& b, const V& i)
{
  for (auto& entry : Dune::Impl::asVector(b))
    entry = i;
}


template<class VectorBlock, class A=std::allocator<void> >
int testVector()
{
  using Alloc = typename std::allocator_traits<A>::template rebind_alloc<VectorBlock>;
  typedef Dune::BlockVector<VectorBlock, Alloc> Vector;

  // empty vector
  Vector v, w, v1(20), v2(20,100);
#ifdef FAIL
  Vector v3(20,100.0);
#endif
  v.reserve(100);
  assert(100==v.capacity());
  assert(20==v1.capacity());
  assert(100==v2.capacity());
  assert(20==v1.N());
  assert(20==v2.N());

  v.resize(25);

  assert(25==v.N());

  for(typename Vector::size_type i=0; i < v.N(); ++i)
    v[i] = i;

  for(typename Vector::size_type i=0; i < v2.N(); ++i)
    v2[i] = i*10;
  w = v;

  testHomogeneousRandomAccessContainer(v);
  Dune::testConstructibility<Vector>();
  testNorms(v);
  testVectorSpaceOperations(v);
  testScalarProduct(v);

  assert(w.N()==v.N());

  for(typename Vector::size_type i=0; i < v.N(); ++i)
    assert(v[i] == w[i]);

  Vector z(w);

  assert(w.N()==z.N());
  assert(w.capacity()==z.capacity());

  for(typename Vector::size_type i=0; i < w.N(); ++i)
    assert(z[i] == w[i]);

  v.reserve(150);
  assert(150==v.capacity());
  assert(25==v.N());

  VectorBlock b;

  // check the entries
  for(typename Vector::size_type i=0; i < v.N(); ++i) {
    assign(b, i);
    assert(v[i] == b);
  }

  return 0;
}

void testCapacity()
{
  typedef Dune::FieldVector<double,2> SmallVector;
  typedef Dune::BlockVector<Dune::BlockVector<SmallVector> > ThreeLevelVector;
  ThreeLevelVector vec;
  vec.reserve(10);
  vec.resize(10);
  for(int i=0; i<10; ++i)
    vec[i]=Dune::BlockVector<SmallVector>(10);
  ThreeLevelVector vec1=vec;
}

template <class V>
void checkNormNAN(V const &v, int line) {
  if (!std::isnan(v.one_norm())) {
    std::cerr << "error: norm not NaN: one_norm() on line "
              << line << " (type: " << Dune::className(v[0][0]) << ")"
              << std::endl;
    std::exit(-1);
  }
  if (!std::isnan(v.two_norm())) {
    std::cerr << "error: norm not NaN: two_norm() on line "
              << line << " (type: " << Dune::className(v[0][0]) << ")"
              << std::endl;
    std::exit(-1);
  }
  if (!std::isnan(v.infinity_norm())) {
    std::cerr << "error: norm not NaN: infinity_norm() on line "
              << line << " (type: " << Dune::className(v[0][0]) << ")"
              << std::endl;
    std::exit(-1);
  }
}

// Make sure that vectors with NaN entries have norm NaN.
// See also bug flyspray/FS#1147
template <typename T>
void
test_nan(T const &mynan)
{
  using FV = Dune::FieldVector<T,2>;
  using V = Dune::BlockVector<FV>;
  T n(0);
  {
    V v = {
      { mynan, n },
      { n, n }
    };
    checkNormNAN(v, __LINE__);
  }
  {
    V v = {
      { n, mynan },
      { n, n }
    };
    checkNormNAN(v, __LINE__);
  }
  {
    V v = {
      { n, n },
      { mynan, n }
    };
    checkNormNAN(v, __LINE__);
  }
  {
    V v = {
      { n, n },
      { n, mynan }
    };
    checkNormNAN(v, __LINE__);
  }
  {
    V v = {
      { mynan, mynan },
      { mynan, mynan }
    };
    checkNormNAN(v, __LINE__);
  }
}

int main()
{
  typedef std::complex<double> value_type;
  //typedef double value_type;
  typedef Dune::FieldVector<value_type,1> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  Vector v;
  v=0;
  Dune::BlockVector<Dune::FieldVector<std::complex<double>,1> > v1;
  v1=0;

  // Test a BlockVector of BlockVectors
  typedef Dune::BlockVector<Vector> VectorOfVector;
  VectorOfVector vv = {{1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0}};

  testHomogeneousRandomAccessContainer(vv);
  Dune::testConstructibility<VectorOfVector>();
  testNorms(vv);
  testVectorSpaceOperations(vv);
  testScalarProduct(vv);

  // Test construction from initializer_list
  Vector fromInitializerList = {0,1,2};
  assert(fromInitializerList.size() == 3);
  assert(fromInitializerList[0] == value_type(0));
  assert(fromInitializerList[1] == value_type(1));
  assert(fromInitializerList[2] == value_type(2));

  {
    double nan = std::nan("");
    test_nan(nan);
  }
  {
    std::complex<double> nan( std::nan(""), 17 );
    test_nan(nan);
  }

  int ret = 0;

  ret += testVector<Dune::FieldVector<double,1> >();
  //  ret += testVector<1, Dune::PoolAllocator<void,1000000> >();
#if HAVE_MPROTECT
  ret += testVector<Dune::FieldVector<double,1> , Dune::DebugAllocator<void> >();
#endif
  ret += testVector<Dune::FieldVector<double,3> >();
  //  ret += testVector<3, Dune::PoolAllocator<void,1000000> >();
#if HAVE_MPROTECT
  ret += testVector<Dune::FieldVector<double,1> , Dune::DebugAllocator<void> >();
#endif

  ret += testVector<double>();
  ret += testVector<std::complex<double> >();

  testCapacity();

  return ret;
}
