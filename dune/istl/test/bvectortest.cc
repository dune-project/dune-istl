// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/common/fvector.hh>
#include <dune/common/poolallocator.hh>
#include <dune/common/debugallocator.hh>

template<typename T, int BS>
void assign(Dune::FieldVector<T,BS>& b, const T& i)
{

  for(int j=0; j < BS; j++)
    b[j] = i;
}


template<int BS, class A=std::allocator<void> >
int testVector()
{

  typedef Dune::FieldVector<int,BS> VectorBlock;
  typedef typename A::template rebind<VectorBlock>::other Alloc;
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


  assert(w.N()==v.N());
  assert(w.capacity()==v.capacity());

  for(typename Vector::size_type i=0; i < v.N(); ++i)
    assert(v[i] == w[i]);

  w = static_cast<const Dune::block_vector_unmanaged<VectorBlock,Alloc>&>(v);

  for(typename Vector::size_type i=0; i < w.N(); ++i)
    assert(v[i] == w[i]);

  Vector z(w);

  assert(w.N()==z.N());
  assert(w.capacity()==z.capacity());

  for(typename Vector::size_type i=0; i < w.N(); ++i)
    assert(z[i] == w[i]);

  Vector z1(static_cast<const Dune::block_vector_unmanaged<VectorBlock,Alloc>&>(v2));

  assert(v2.N()==z1.N());
  assert(v2.capacity()==z1.capacity());

  for(typename Vector::size_type i=1; i < v2.N(); ++i) {
    assert(z1[i] == v2[i]);
  }

  v.reserve(150);
  assert(150==v.capacity());
  assert(25==v.N());

  VectorBlock b;

  // check the entries
  for(typename Vector::size_type i=0; i < v.N(); ++i) {
    assign(b, (int)i);
    assert(v[i] == b);
  }

  // Try to shrink the vector
  v.reserve(v.N());

  assert(v.N()==v.capacity());

  // check the entries

  for(typename Vector::size_type i=0; i < v.N(); ++i) {
    assign(b,(int)i);
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
  vec.reserve(20, true);
  vec.reserve(10, true);
  vec.reserve(5, false);
  vec.reserve(20, false);
  vec.reserve(0, true);
  vec1.reserve(0, false);
}

int main()
{
  typedef std::complex<double> value_type;
  //typedef double value_type;
  typedef Dune::FieldVector<value_type,1> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::BlockVector<Vector> VectorOfVector;
  Vector v;
  v=0;
  Dune::BlockVector<Dune::FieldVector<std::complex<double>,1> > v1;
  v1=0;
  VectorOfVector vv;
  vv.two_norm();

  int ret = 0;

  ret += testVector<1>();
  //  ret += testVector<1, Dune::PoolAllocator<void,1000000> >();
  ret += testVector<1, Dune::DebugAllocator<void> >();
  ret += testVector<3>();
  //  ret += testVector<3, Dune::PoolAllocator<void,1000000> >();
  ret += testVector<3, Dune::DebugAllocator<void> >();

  testCapacity();

  return ret;
}
