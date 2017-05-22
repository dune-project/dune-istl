// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/common/fmatrix.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/istl/scaledidmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include "laplacian.hh"

/*  "tests" the writeMatrixToMatlabHelper method by calling it for a Laplacian with a given BlockType and writing to cout.
 *  Actual functionality is not tested.
 */
template <class BlockType>
void testWriteMatrixToMatlab(BlockType b=BlockType(0.0))
{
  typedef Dune::BCRSMatrix<BlockType> Matrix;

  Matrix A;
  setupLaplacian(A, 3);

  A[0][0] += b;
  writeMatrixToMatlabHelper(A, 0, 0, std::cout);
}

/* uses the writeVectorToMatlab method, filled with dummy data */
template <class VectorType>
void testWriteVectorToMatlab()
{
  VectorType v;
  for (unsigned int i = 0; i < v.size(); ++i)
  {
    v[i] = i;
  }

  Dune::writeVectorToMatlabHelper(v, std::cout);
}

int main(int argc, char** argv)
{
  /* testing the writeMatrixToMatlabHelper method for BlockType=FieldMatrix with different field_types */
  testWriteMatrixToMatlab<Dune::FieldMatrix<double,1,1> >();
  testWriteMatrixToMatlab<Dune::FieldMatrix<double,1,2> >();
  //    testWriteMatrixToMatlab<Dune::FieldMatrix<double,2,1> >(); // commented because setUpLaplacian cannot handle block_types with more rows than cols
  testWriteMatrixToMatlab<Dune::FieldMatrix<double,4,7> >();
  //    testWriteMatrixToMatlab<Dune::FieldMatrix<double,7,4> >(); // commented because setUpLaplacian cannot handle block_types with more rows than cols
  testWriteMatrixToMatlab<Dune::FieldMatrix<double,2,2> >();

  testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,1,1> >(Dune::FieldMatrix<std::complex<double>,1,1>(std::complex<double>(0, 1)));
  testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,1,2> >(Dune::FieldMatrix<std::complex<double>,1,2>(std::complex<double>(0, 1)));
  //    testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,2,1> >(); // commented because setUpLaplacian cannot handle block_types with more rows than cols
  testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,4,7> >(Dune::FieldMatrix<std::complex<double>,4,7>(std::complex<double>(0, 1)));
  //    testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,7,4> >(); // commented because setUpLaplacian cannot handle block_types with more rows than cols
  testWriteMatrixToMatlab<Dune::FieldMatrix<std::complex<double>,2,2> >(Dune::FieldMatrix<std::complex<double>,2,2>(std::complex<double>(0, 1)));

  /* testing the writeMatrixToMatlabHelper method for BlockType=[Diagonal|ScaledIdentity]Matrix with different field_types */
  testWriteMatrixToMatlab<Dune::DiagonalMatrix<double,1> >();
  testWriteMatrixToMatlab<Dune::ScaledIdentityMatrix<double,1> >();
  testWriteMatrixToMatlab<Dune::DiagonalMatrix<double,2> >();
  testWriteMatrixToMatlab<Dune::ScaledIdentityMatrix<double,2> >();

  testWriteMatrixToMatlab<Dune::DiagonalMatrix<std::complex<double>,1> >(Dune::DiagonalMatrix<std::complex<double>,1>(std::complex<double>(0,1)));
  testWriteMatrixToMatlab<Dune::ScaledIdentityMatrix<std::complex<double>,1> >(Dune::ScaledIdentityMatrix<std::complex<double>,1>(std::complex<double>(0,1)));
  testWriteMatrixToMatlab<Dune::DiagonalMatrix<std::complex<double>,2> >(Dune::DiagonalMatrix<std::complex<double>,2>(std::complex<double>(0,1)));
  testWriteMatrixToMatlab<Dune::ScaledIdentityMatrix<std::complex<double>,2> >(Dune::ScaledIdentityMatrix<std::complex<double>,2>(std::complex<double>(0,1)));


  /* testing the test writeMatrixToMatlabHelper for FieldVector */
  testWriteVectorToMatlab<Dune::FieldVector<double,1> >();
  testWriteVectorToMatlab<Dune::FieldVector<float,5> >();
  testWriteVectorToMatlab<Dune::FieldVector<std::complex<double>,5> >();
  /* testing the test writeMatrixToMatlabHelper for BlockVector */
  Dune::BlockVector<Dune::FieldVector<double,3> > v1 = {{1.0, 2.0, 3.0}};
  Dune::writeVectorToMatlabHelper(v1, std::cout);
  Dune::BlockVector<Dune::BlockVector<Dune::FieldVector<double,1> > > v2 = {{1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0}};
  Dune::writeVectorToMatlabHelper(v2, std::cout);
  /* testing the test writeMatrixToMatlabHelper for STL containers */
  testWriteVectorToMatlab<std::array<double,5> >();
  std::vector<double> v3;
  v3.push_back(1);
  v3.push_back(2);
  Dune::writeVectorToMatlabHelper(v3, std::cout);
}
