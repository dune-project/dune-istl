
#include <config.h>

#include <iostream>

#include <dune/common/test/testsuite.hh>
#include <dune/common/indices.hh>

#include <dune/istl/sparseforeach.hh>

using namespace Dune;

TestSuite testSparseForEach()
{
  TestSuite t;

  FieldVector<double,3> f3;
  FieldVector<double,1> f1;

  BlockVector<FieldVector<double,3>> b3;
  BlockVector<FieldVector<double,1>> b1;

  b3.resize(5);
  b1.resize(5);

  using MTBV = MultiTypeBlockVector<BlockVector<FieldVector<double,3>>,BlockVector<FieldVector<double,1>>>;

  MTBV v;

  v[Indices::_0] = b3;
  v[Indices::_1] = b1;

  int entries = 0;

  auto countEntres = [&](auto&& index, auto&& entry){
    std::cout << "index = " << index << " entry = " << entry << std::endl;
    entries++;
  };

  FlatVectorView flatView(v);

  sparseForEach(flatView,countEntres);

  t.check( entries == 20 );
  return t;
}



int main(int argc, char** argv)
{
  TestSuite t;

  t.subTest(testSparseForEach());

  return t.exit();
}
