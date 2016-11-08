// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/common/fvector.hh>
#include <dune/common/poolallocator.hh>
#include <dune/common/debugallocator.hh>
#include <dune/common/classname.hh>

#include <dune/istl/test/vectortest.hh>

template<typename T, int BS>
void assign(Dune::FieldVector<T,BS>& b, const T& i)
{

  for(int j=0; j < BS; j++)
    b[j] = i;
}


template<int BS, class A=std::allocator<void> >
int testVector()
{

  typedef Dune::FieldVector<double,BS> VectorBlock;
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

  testHomogeneousRandomAccessContainer(v);
  Dune::testConstructibility<Vector>();
  testNorms(v);
  testVectorSpaceOperations(v);
  testScalarProduct(v);

  assert(w.N()==v.N());
  assert(w.capacity()==v.capacity());

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
    assign(b, (typename VectorBlock::field_type)i);
    assert(v[i] == b);
  }

  // Try to shrink the vector
  v.reserve(v.N());

  assert(v.N()==v.capacity());

  // check the entries

  for(typename Vector::size_type i=0; i < v.N(); ++i) {
    assign(b,(typename VectorBlock::field_type)i);
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

  ret += testVector<1>();
  //  ret += testVector<1, Dune::PoolAllocator<void,1000000> >();
  ret += testVector<1, Dune::DebugAllocator<void> >();
  ret += testVector<3>();
  //  ret += testVector<3, Dune::PoolAllocator<void,1000000> >();
  ret += testVector<3, Dune::DebugAllocator<void> >();

  testCapacity();

  return ret;
}
