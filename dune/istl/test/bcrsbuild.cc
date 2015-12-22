// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>
#include <dune/istl/io.hh>

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
  void randomBuild(int rows, int cols)
  {
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
  }

  void rowWiseBuild(Dune::BCRSMatrix<B,A>& matrix, int /* rows */, int cols)
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

    Dune::printmatrix(std::cout, matrix, "row_wise", "row");
    // test copy ctor
    Dune::BCRSMatrix<B,A> matrix1(matrix);
    Dune::printmatrix(std::cout, matrix1, "row_wise", "row");
    // test copy assignment
    Dune::BCRSMatrix<B,A> matrix2;
    matrix2 = matrix;
    Dune::printmatrix(std::cout, matrix2, "row_wise", "row");
  }

  void rowWiseBuild(int rows, int cols)
  {
    Dune::BCRSMatrix<B,A> matrix( rows, cols, Dune::BCRSMatrix<B,A>::row_wise );
    rowWiseBuild(matrix, rows, cols);
  }

  void rowWiseBuild(int rows, int cols, int nnz)
  {
    Dune::BCRSMatrix<B,A> matrix( rows, cols, nnz, Dune::BCRSMatrix<B,A>::row_wise );
    rowWiseBuild(matrix, rows, cols);
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
  try{
    Builder<Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > > builder;
    builder.randomBuild(5,4);
    builder.rowWiseBuild(5,4,13);
    builder.rowWiseBuild(5,4);
    testDoubleSetSize();
  }catch(Dune::Exception e) {
    std::cerr << e<<std::endl;
    return 1;
  }
}
