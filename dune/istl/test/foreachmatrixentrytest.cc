
#include <config.h>

#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/foreachmatrixentry.hh>
#include <dune/istl/matrixindexset.hh>




using namespace Dune;

TestSuite testTupleSplitter()
{
  TestSuite t;

  auto tuple = std::make_tuple( 0 , 1, Indices::_2 );

  auto tail = Detail::tupleTail(tuple);
  auto tuple2 = std::make_tuple( 1, Indices::_2 );

  t.check( tuple2 == tail );
  return t;
}



template<class Index>
void printIndex( std::ostream& os, const Index& index )
{
  if constexpr ( std::tuple_size_v<Index> > 0 )
  {
    auto head = Detail::tupleHead(index);
    auto tail = Detail::tupleTail(index);
    os << head << " ";
    printIndex( os, tail );
  }
}

TestSuite testRowWiseForEach()
{
  TestSuite t;

  FieldMatrix<double,3,3> F33;
  FieldMatrix<double,3,1> F31;
  FieldMatrix<double,1,3> F13;
  FieldMatrix<double,1,1> F11;

  BCRSMatrix<FieldMatrix<double,3,3>> B33;
  BCRSMatrix<FieldMatrix<double,3,1>> B31;
  BCRSMatrix<FieldMatrix<double,1,3>> B13;
  BCRSMatrix<FieldMatrix<double,1,1>> B11;

  MatrixIndexSet mis;
  mis.resize(3,3);

  // set some indices ( skip one row for the top left block )
  mis.add(0,0);
  mis.add(2,1);

  mis.exportIdx(B33);

  mis.add(1,1);

  mis.exportIdx(B31);
  mis.exportIdx(B13);
  mis.exportIdx(B11);

  using Row0 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,3,3>>,BCRSMatrix<FieldMatrix<double,3,1>>>;
  using Row1 = MultiTypeBlockVector<BCRSMatrix<FieldMatrix<double,1,3>>,BCRSMatrix<FieldMatrix<double,1,1>>>;

  using MTMatrix = MultiTypeBlockMatrix<Row0,Row1>;

  MTMatrix M;
  M[Indices::_0][Indices::_0] = B33;
  M[Indices::_0][Indices::_1] = B31;
  M[Indices::_1][Indices::_0] = B13;
  M[Indices::_1][Indices::_1] = B11;


  int entries = 0;
  int rows = 0;

  auto entryCounter = [&](auto&& rowIdx, auto&& colIdx, auto&& /*entry*/){
    entries++;
    std::cout << "entryCounter with row = ";
    printIndex( std::cout, rowIdx );
    std::cout << " and col = ";
    printIndex( std::cout, colIdx );
    std::cout << "\n";
  };

  auto rowCounter = [&](auto&& /*index*/)
  {
    rows++;
  };

  forEachScalarMatrixEntry(M,entryCounter);
  forEachScalarMatrixRow(M,rowCounter);

  t.check( entries == 39 );
  t.check( rows == 12 );
  return t;
}



int main(int argc, char** argv)
{
  TestSuite t;

  t.subTest(testTupleSplitter());
  t.subTest(testRowWiseForEach());

  return t.exit();
}
