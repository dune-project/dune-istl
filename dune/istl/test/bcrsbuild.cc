// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/istl/io.hh>

template<class M>
std::size_t computeNNZ(M&& matrix)
{
  std::size_t nnz = 0;
  for (auto&& row : matrix)
    for ([[maybe_unused]] auto&& entry : row)
      ++nnz;
  return nnz;
}


template<class M>
struct Builder
{
  void randomBuild(int m, int n)
  {
    DUNE_THROW(Dune::NotImplemented, "No specialization");
  }
};


template<class B, class A>
struct Builder<Dune::BCRSMatrix<B,A> >
{
  Dune::TestSuite randomBuild(int rows, int cols)
  {
    Dune::TestSuite testSuite("BCRSMatrix::random");

    int maxNZCols = 15; // maximal number of nonzeros per row
    {
      Dune::BCRSMatrix<B,A> matrix( rows, cols, Dune::BCRSMatrix<B,A>::random );
      for(int i=0; i<rows; ++i)
        matrix.setrowsize(i,maxNZCols);
      matrix.endrowsizes();

      // During setup, or before, if there is no other way.
      for(int i=0; i<rows; ++i) {
        if(i<cols)
          matrix.addindex(i,i);
        if(i-1>=0)
          matrix.addindex(i,i-1);
        if(i+1<cols)
          matrix.addindex(i,i+1);
      }

      matrix.endindices();
      matrix=1;

      Dune::printmatrix(std::cout, matrix, "random", "row");
      Dune::BCRSMatrix<B,A> matrix1(matrix);
      Dune::printmatrix(std::cout, matrix, "random", "row");

      auto nnz = computeNNZ(matrix);
      testSuite.check(nnz == matrix.nonzeroes(), "nnz after building")
        << "BCRSMatrix::nonzeroes() returns " << matrix.nonzeroes() << " while there are " << nnz << " stored entries.";
    }
    /*{

       Dune::BCRSMatrix<B,A> matrix( rows, cols, rows*maxNZCols, Dune::BCRSMatrix<B,A>::random );
       for(int i=0; i<rows; ++i)
         matrix.setrowsize(i,maxNZCols);
       matrix.endrowsizes();

       for(int i=0; i<rows; ++i){
        if(i<cols)
          matrix.addindex(i,i);
        if(i-1>=0)
          matrix.addindex(i,i-1);
        if(i+1<cols)
          matrix.addindex(i,i+1);
       }
       matrix.endrowsizes();

       Dune::printmatrix(std::cout, matrix, "random", "row");
       }*/
    return testSuite;
  }

  void rowWiseBuild(Dune::TestSuite& testSuite, Dune::BCRSMatrix<B,A>& matrix, int /* rows */, int cols)
  {
    for(typename Dune::BCRSMatrix<B,A>::CreateIterator ci=matrix.createbegin(), cend=matrix.createend();
        ci!=cend; ++ci)
    {
      int i=ci.index();
      if(i<cols)
        ci.insert(i);
      if(i-1>=0 && i-1<cols)
        ci.insert(i-1);
      if(i+1<cols)
        ci.insert(i+1);
    }

    {
      auto nnz = computeNNZ(matrix);
      testSuite.check(nnz == matrix.nonzeroes(), "nnz after building")
        << "BCRSMatrix::nonzeroes() returns " << matrix.nonzeroes() << " while there are " << nnz << " stored entries.";
    }

    Dune::printmatrix(std::cout, matrix, "row_wise", "row");
    // test copy ctor
    Dune::BCRSMatrix<B,A> matrix1(matrix);
    Dune::printmatrix(std::cout, matrix1, "row_wise", "row");

    {
      auto nnz = computeNNZ(matrix1);
      testSuite.check(nnz == matrix1.nonzeroes(), "nnz after constructed from")
        << "BCRSMatrix::nonzeroes() returns " << matrix1.nonzeroes() << " while there are " << nnz << " stored entries.";
    }

    // test copy assignment
    Dune::BCRSMatrix<B,A> matrix2;
    matrix2 = matrix;
    Dune::printmatrix(std::cout, matrix2, "row_wise", "row");

    {
      auto nnz = computeNNZ(matrix2);
      testSuite.check(nnz == matrix2.nonzeroes(), "nnz after assigned from")
        << "BCRSMatrix::nonzeroes() returns " << matrix2.nonzeroes() << " while there are " << nnz << " stored entries.";
    }
  }

  Dune::TestSuite rowWiseBuild(int rows, int cols)
  {
    Dune::TestSuite testSuite("BCRSMatrix::row_wise without nnz");
    Dune::BCRSMatrix<B,A> matrix( rows, cols, Dune::BCRSMatrix<B,A>::row_wise );
    rowWiseBuild(testSuite, matrix, rows, cols);
    return testSuite;
  }

  Dune::TestSuite rowWiseBuild(int rows, int cols, int nnz)
  {
    Dune::TestSuite testSuite("BCRSMatrix::row_wise with nnz");
    Dune::BCRSMatrix<B,A> matrix( rows, cols, nnz, Dune::BCRSMatrix<B,A>::row_wise );
    rowWiseBuild(testSuite, matrix, rows, cols);
    return testSuite;
  }

};

// This code used to trigger a valgrind 'uninitialized memory' warning; see FS 1041
void testDoubleSetSize()
{
  Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > foo;
  foo.setSize(5,5);
  foo.setSize(5,5);
}


int main()
{
  Dune::TestSuite testSuite;

  Builder<Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > > builder;

  testSuite.subTest(builder.randomBuild(5,4));
  testSuite.subTest(builder.rowWiseBuild(5,4,13));
  testSuite.subTest(builder.rowWiseBuild(5,4));

  testDoubleSetSize();

  return testSuite.exit();
}
