// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include <config.h>

#include <iostream>

#include <dune/common/bitsetvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/tuplevector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/matrixindexset.hh>




using namespace Dune;

TestSuite testFlatVectorForEach()
{
  TestSuite t;

  // mix up some types

  [[maybe_unused]] FieldVector<double,3> f3;
  [[maybe_unused]] FieldVector<double,1> f1;

  DynamicVector<FieldVector<double,3>> d3;

  std::vector<FieldVector<double,1>> v1;

  d3.resize(5);
  v1.resize(5);

  using MTBV = MultiTypeBlockVector<DynamicVector<FieldVector<double,3>>,std::vector<FieldVector<double,1>>>;

  MTBV v;

  v[Indices::_0] = d3;
  v[Indices::_1] = v1;

  int entries = 0;

  auto countEntres = [&](auto&& entry, auto&& index){
    entries++;
  };

  auto s = flatVectorForEach(v,countEntres);

  t.check( entries == 20 );
  t.check( s == 20 );

  return t;
}


TestSuite testFlatVectorForEachBitSetVector()
{
  TestSuite t;

  int entries = 0;

  auto countEntres = [&](auto&& entry, auto&& index){
    entries++;
  };

  BitSetVector<2> bitSetVector;
  bitSetVector.resize(10);

  auto s = flatVectorForEach(bitSetVector,countEntres);

  t.check( entries == 20 );
  t.check( s == 20 );

  return t;
}


TestSuite testFlatMatrixForEachStatic()
{
  TestSuite t;

  [[maybe_unused]] FieldMatrix<double,3,3> F33;
  [[maybe_unused]] FieldMatrix<double,3,1> F31;
  [[maybe_unused]] FieldMatrix<double,1,3> F13;
  [[maybe_unused]] FieldMatrix<double,1,1> F11;

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

  auto [ rows , cols ] = flatMatrixForEach( M, [&](auto&& /*entry*/, auto&& rowIndex, auto&& colIndex){

    entries++;

  });

  t.check( entries == 39 , " wrong number of entries ");
  t.check( rows == 12 , " wrong number of rows ");
  t.check( cols == 12 , " wrong number of cols ");
  return t;
}



TestSuite testFlatMatrixForEachDynamic()
{
  TestSuite t;

  DynamicMatrix<double> F33(3,3);


  BCRSMatrix<DynamicMatrix<double>> B;


  MatrixIndexSet mis;
  mis.resize(3,3);

  // set some entries and leave one line empty on purpose
  mis.add(0,0);
  mis.add(1,1);

  mis.exportIdx(B);

  B[0][0] = F33;
  B[1][1] = F33;


  int entries = 0;

  auto [ rows , cols ] = flatMatrixForEach( B, [&](auto&& /*entry*/, auto&& rowIndex, auto&& colIndex){

    entries++;

  });

  t.check( entries == 18 , " wrong number of entries ");
  t.check( rows == 9 , " wrong number of rows ");
  t.check( cols == 9 , " wrong number of cols ");
  return t;
}



int main(int argc, char** argv)
{
  TestSuite t;

  t.subTest(testFlatVectorForEach());
  t.subTest(testFlatVectorForEachBitSetVector());
  t.subTest(testFlatMatrixForEachStatic());
  t.subTest(testFlatMatrixForEachDynamic());

  return t.exit();
}
