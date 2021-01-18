
#include <config.h>

#include <iostream>

#include <dune/common/test/testsuite.hh>
#include <dune/common/indices.hh>

#include <dune/istl/vectoriterator.hh>

using namespace Dune;

template<class Index>
void printIndex( std::ostream& os, const Index& index )
{
  if constexpr ( std::tuple_size_v<Index> > 0 )
  {
    auto head = Impl::tupleHead(index);
    auto tail = Impl::tupleTail(index);
    os << head << " ";
    printIndex( os, tail );
  }
}

TestSuite testVectorIterator()
{
  TestSuite t;

  FieldVector<double,3> F3;
  FieldVector<double,1> F1;

  BlockVector<FieldVector<double,3>> B3;
  BlockVector<FieldVector<double,1>> B1;

  B3.resize(5);
  B1.resize(5);

  using MTBV = MultiTypeBlockVector<BlockVector<FieldVector<double,3>>,BlockVector<FieldVector<double,1>>>;

  MTBV v;

  v[Indices::_0] = B3;
  v[Indices::_1] = B1;

  int entries = 0;

  auto entryCounter = [&](auto&& index, auto&& /*entry*/){
    entries++;
    std::cout << "entryAction with index = ";
    printIndex( std::cout, index );
    std::cout << "\n";
  };

  VectorIterator vi(v);
  vi.iterate( entryCounter, std::tuple<>() );

  t.check( entries == 20 );
  return t;
}



int main(int argc, char** argv)
{
  TestSuite t;

  t.subTest(testVectorIterator());

  return t.exit();
}
