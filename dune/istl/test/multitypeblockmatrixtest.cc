// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \file
 * \brief Test the MultiTypeBlockMatrix data structure
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>

using namespace Dune;

int main(int argc, char** argv) try
{
#if ! HAVE_DUNE_BOOST || ! defined HAVE_BOOST_FUSION
  return 77;
#else
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //  First, we test a MultiTypeBlockMatrix consisting of an array of 2x2 dense matrices.
  //  The upper left dense matrix has dense 3x3 blocks, the lower right matrix has 1x1 blocks,
  //  the off-set diagonal matrix block sizes are set accordingly.
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // set up the test matrix
  typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,3,3> >, Matrix<FieldMatrix<double,3,1> > > RowType0;
  typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,1,3> >, Matrix<FieldMatrix<double,1,1> > > RowType1;

  std::integral_constant<int, 0> _0;
  std::integral_constant<int, 1> _1;

  MultiTypeBlockMatrix<RowType0,RowType1> multiMatrix;

  fusion::at_c<0>(multiMatrix)[_0].setSize(3,3);
  fusion::at_c<0>(multiMatrix)[_1].setSize(3,2);
  fusion::at_c<1>(multiMatrix)[_0].setSize(2,3);
  fusion::at_c<1>(multiMatrix)[_1].setSize(2,2);

  // lazy solution: initialize the entire matrix with zeros
  fusion::at_c<0>(multiMatrix)[_0] = 0;
  fusion::at_c<0>(multiMatrix)[_1] = 0;
  fusion::at_c<1>(multiMatrix)[_0] = 0;
  fusion::at_c<1>(multiMatrix)[_1] = 0;

  printmatrix(std::cout, fusion::at_c<0>(multiMatrix)[_0], "(0,0)", "--");
  printmatrix(std::cout, fusion::at_c<0>(multiMatrix)[_1], "(0,1)", "--");
  printmatrix(std::cout, fusion::at_c<1>(multiMatrix)[_0], "(1,0)", "--");
  printmatrix(std::cout, fusion::at_c<1>(multiMatrix)[_1], "(1,1)", "--");

  // set up a test vector
  MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > > multiVector;

  multiVector[_0] = {{1,0,0},
                     {0,1,0},
                     {0,0,1}};

  multiVector[_1] = {3.14, 42};

  // Test matrix-vector products
  MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > > result;
  result[_0].resize(3);
  result[_1].resize(2);

  multiMatrix.mv(multiVector,result);
  multiMatrix.umv(multiVector,result);
  multiMatrix.mmv(multiVector,result);
  multiMatrix.usmv(3.14,multiVector,result);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //  Next test: Set up a linear system with a matrix consisting of 2x2 sparse scalar matrices.
  //  Solve the linear system with a simple iterative scheme.
  //  \todo What's the point if all four matrices have the same type?
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Jacobi Solver Test on MultiTypeBlockMatrix<BCRSMatrix>";

  typedef Dune::FieldMatrix<double,1,1> LittleBlock;                    //matrix block type
  typedef Dune::BCRSMatrix<LittleBlock> BCRSMat;                        //matrix type

  const int X1=3;                                                       //index bounds of all four matrices
  const int X2=2;
  const int Y1=3;
  const int Y2=2;
  BCRSMat A11 = BCRSMat(X1,Y1,X1*Y1,BCRSMat::random);                   //A11 is 3x3
  BCRSMat A12 = BCRSMat(X1,Y2,X1*Y2,BCRSMat::random);                   //A12 is 2x3
  BCRSMat A21 = BCRSMat(X2,Y1,X2*Y1,BCRSMat::random);                   //A11 is 3x2
  BCRSMat A22 = BCRSMat(X2,Y2,X2*Y2,BCRSMat::random);                   //A12 is 2x2

  typedef Dune::MultiTypeBlockVector<Dune::BlockVector<Dune::FieldVector<double,1> >,Dune::BlockVector<Dune::FieldVector<double,1> > > TestVector;
  TestVector x, b;

  x[_0].resize(Y1);
  x[_1].resize(Y2);
  b[_0].resize(X1);
  b[_1].resize(X2);

  x = 1; b = 1;

  //set row sizes
  for (int i=0; i<Y1; i++)
  {
    A11.setrowsize(i,X1);
    A12.setrowsize(i,X2);
  }
  A11.endrowsizes();
  A12.endrowsizes();

  for (int i=0; i<Y2; i++)
  {
    A21.setrowsize(i,X1);
    A22.setrowsize(i,X2);
  }
  A21.endrowsizes();
  A22.endrowsizes();

  //set indices
  for (int i=0; i<X1+X2; i++)
  {
    for (int j=0; j<Y1+Y2; j++)
    {
      if (i<X1 && j<Y1)
        A11.addindex(i,j);
      if (i<X1 && j>=Y1)
        A12.addindex(i,j-Y1);
      if (i>=X1 && j<Y1)
        A21.addindex(i-X1,j);
      if (i>=X1 && j>=Y1)
        A22.addindex(i-X1,j-Y1);
    }
  }
  A11.endindices();
  A12.endindices();
  A21.endindices();
  A22.endindices();

  A11 = 0;
  A12 = 0;
  A21 = 0;
  A22 = 0;

  //fill in values (row-wise) in A11 and A22
  for (int i=0; i<Y1; i++)
  {
    if (i>0)
      A11[i][i-1]=-1;
    A11[i][i]=2;                                                        //diag
    if (i<Y1-1)
      A11[i][i+1]=-1;
    if (i<Y2)
    {                                                         //also in A22
      if (i>0)
        A22[i][i-1]=-1;
      A22[i][i]=2;
      if (i<Y2-1)
        A22[i][i+1]=-1;
    }
  }
  A12[2][0] = -1; A21[0][2] = -1;                                       //fill A12 & A21

  typedef Dune::MultiTypeBlockVector<BCRSMat,BCRSMat> BCRS_Row;
  typedef Dune::MultiTypeBlockMatrix<BCRS_Row,BCRS_Row> CM_BCRS;
  CM_BCRS A;
  fusion::at_c<0>(A)[_0] = A11;
  fusion::at_c<0>(A)[_1] = A12;
  fusion::at_c<1>(A)[_0] = A21;
  fusion::at_c<1>(A)[_1] = A22;

  x = 1;
  b = 1;

  // Set up a variety of solvers, just to make sure they compile
  MatrixAdapter<CM_BCRS,TestVector,TestVector> op(A);             // make linear operator from A
  SeqJac<CM_BCRS,TestVector,TestVector,2> jac(A,1,1);                // Jacobi preconditioner
  SeqGS<CM_BCRS,TestVector,TestVector,2> gs(A,1,1);                  // GS preconditioner
  SeqSOR<CM_BCRS,TestVector,TestVector,2> sor(A,1,1.9520932);        // SOR preconditioner
  SeqSSOR<CM_BCRS,TestVector,TestVector,2> ssor(A,1,1.0);      // SSOR preconditioner
  LoopSolver<TestVector> loop(op,gs,1E-4,18000,2);           // an inverse operator
  InverseOperatorResult r;

  // Solve linear system
  loop.apply(x,b,r);

  printvector(std::cout,x[_0],"solution x1","entry",11,9,1);
  printvector(std::cout,x[_1],"solution x2","entry",11,9,1);

  return 0;
#endif
}
catch (Dune::Exception& e)
{
  std::cerr << "DUNE reported an exception: " << e << std::endl;
  return 1;
}
catch (std::exception& e)
{
  std::cerr << "C++ reported an exception: " << e.what() << std::endl;
  return 2;
} catch (...)
{
  std::cerr << "Unknown exception encountered!" << std::endl;
  return 3;
}