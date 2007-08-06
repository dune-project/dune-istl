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
    int maxNZCols = 15; // maximal number of nonzeros per column
    {

      Dune::BCRSMatrix<B,A> matrix( rows, cols, Dune::BCRSMatrix<B,A>::random );
      for(int i=0; i<rows; ++i) matrix.setrowsize(i,maxNZCols);
      matrix.endrowsizes();

      ////////////////////////
      //während des Aufstellens
      // oder wenn es nicht anders geht auch davor.
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
    }
    /*{

       Dune::BCRSMatrix<B,A> matrix( rows, cols, rows*maxNZCols, Dune::BCRSMatrix<B,A>::random );
       for(int i=0; i<rows; ++i){
        matrix.setrowsize(i,maxNZCols);
        if(i<cols)
          matrix.addindex(i,i);
        if(i-1>=0)
          matrix.addindex(i,i-1);
        if(i+1<cols)
          matrix.addindex(i,i+1);
       }
       matrix.endrowsizes();

       Dune::printmatrix(std::cout, matrix, "random", "row");
       }
       {

       Dune::BCRSMatrix<B,A> matrix( rows, cols, Dune::BCRSMatrix<B,A>::random );
       for(int i=0; i<rows; ++i){
        matrix.setrowsize(i,maxNZCols);
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
};

int main()
{
  try{
    Builder<Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > > builder;
    builder.randomBuild(5,4);
  }catch(Dune::Exception e) {
    std::cerr << e<<std::endl;
  }
}
