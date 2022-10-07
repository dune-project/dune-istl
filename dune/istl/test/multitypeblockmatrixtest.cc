// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
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
#include <dune/common/indices.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/test/matrixtest.hh>

using namespace Dune;

// Import the static constants _0, _1, etc
using namespace Indices;

// Test whether we can use std::tuple_element
template<typename... Args>
void testTupleElement(const MultiTypeBlockMatrix<Args...>& multiMatrix)
{
  // Do std::tuple_element and operator[] return the same type?
  using TupleElementType = typename std::tuple_element<0, MultiTypeBlockMatrix<Args...> >::type;
  using BracketType = decltype(multiMatrix[_0]);

  // As the return type of const operator[], BracketType will always
  // be a const reference.  We cannot simply strip the const and the &,
  // because entries of a MultiTypeBlockMatrix can be references themselves.
  // Therefore, always add const& to the result of std::tuple_element as well.
  constexpr bool sameType = std::is_same_v<const TupleElementType&,BracketType>;
  static_assert(sameType, "std::tuple_element does not provide the type of the 0th MultiTypeBlockMatrix row!");
}

void testInterfaceMethods()
{
  {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  First, we test a MultiTypeBlockMatrix consisting of an array of 2x2 dense matrices.
    //  The upper left dense matrix has dense 3x3 blocks, the lower right matrix has 1x1 blocks,
    //  the off-set diagonal matrix block sizes are set accordingly.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // set up the test matrix
    typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,3,3> >, Matrix<FieldMatrix<double,3,1> > > RowType0;
    typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,1,3> >, Matrix<FieldMatrix<double,1,1> > > RowType1;

    MultiTypeBlockMatrix<RowType0,RowType1> multiMatrix;

    multiMatrix[_0][_0].setSize(3,3);
    multiMatrix[_0][_1].setSize(3,2);
    multiMatrix[_1][_0].setSize(2,3);
    multiMatrix[_1][_1].setSize(2,2);

    // Init with scalar values
    multiMatrix[_0][_0] = 4200;
    multiMatrix[_0][_1] = 4201;
    multiMatrix[_1][_0] = 4210;
    multiMatrix[_1][_1] = 4211;

    // Set up a test vector
    MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<FieldVector<double,1> > > domainVector;

    domainVector[_0] = {{1,0,0},
                        {0,1,0},
                        {0,0,1}};

    domainVector[_1] = {3.14, 42};

    auto rangeVector = domainVector;   // Range == Domain, because the matrix nesting pattern is symmetric

    // Test vector space operations
    testVectorSpaceOperations(multiMatrix);

    // Test matrix norms
    testNorms(multiMatrix);

    // Test whether matrix class has the required constructors
    testMatrixConstructibility<decltype(multiMatrix)>();

    // Test matrix-vector products
    testMatrixVectorProducts(multiMatrix, domainVector, rangeVector);

    // Test whether std::tuple_element works
    testTupleElement(multiMatrix);
  }

  {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  We again test a MultiTypeBlockMatrix consisting of an array of 2x2 sparse matrices.
    //  The upper left matrix has dense 3x3 blocks, the lower right matrix has scalar entries
    //  the off-set diagonal matrix block sizes are set accordingly.
    //  Unlike in the previous test, the scalar entries in the lower-right block are actually
    //  scalars, not FieldMatrices of size 1x1.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // set up the test matrix
    typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,3,3> >, Matrix<FieldMatrix<double,3,1> > > RowType0;
    typedef MultiTypeBlockVector<Matrix<FieldMatrix<double,1,3> >, Matrix<double> > RowType1;

    MultiTypeBlockMatrix<RowType0,RowType1> multiMatrix;

    multiMatrix[_0][_0].setSize(3,3);
    multiMatrix[_0][_1].setSize(3,2);
    multiMatrix[_1][_0].setSize(2,3);
    multiMatrix[_1][_1].setSize(2,2);

    // Init with scalar values
    multiMatrix[_0][_0] = 4200;
    multiMatrix[_0][_1] = 4201;
    multiMatrix[_1][_0] = 4210;
    multiMatrix[_1][_1] = 4211;

    // Set up a test vector
    MultiTypeBlockVector<BlockVector<FieldVector<double,3> >, BlockVector<double> > domainVector;

    domainVector[_0] = {{1,0,0},
                        {0,1,0},
                        {0,0,1}};

    domainVector[_1] = {3.14, 42};

    auto rangeVector = domainVector;   // Range == Domain, because the matrix nesting pattern is symmetric

    // Test vector space operations
    testVectorSpaceOperations(multiMatrix);

    // Test whether matrix class has the required constructors
    testMatrixConstructibility<decltype(multiMatrix)>();

    // Test matrix-vector products
    testMatrixVectorProducts(multiMatrix, domainVector, rangeVector);
  }
}

int main(int argc, char** argv) try
{
  // Run the standard tests for the dune-istl matrix interface
  testInterfaceMethods();

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
  A[_0][_0] = A11;
  A[_0][_1] = A12;
  A[_1][_0] = A21;
  A[_1][_1] = A22;

  x = 1;
  b = 1;

  // Set up a variety of solvers, just to make sure they compile
  MatrixAdapter<CM_BCRS,TestVector,TestVector> op(A);             // make linear operator from A
  SeqSOR<CM_BCRS,TestVector,TestVector,2> sor(A,1,1.9520932);        // SOR preconditioner
  SeqSSOR<CM_BCRS,TestVector,TestVector,2> ssor(A,1,1.0);      // SSOR preconditioner

  // Solve system using a Gauss-Seidel method
  SeqGS<CM_BCRS,TestVector,TestVector,2> gs(A,1,1);                  // GS preconditioner
  LoopSolver<TestVector> loop(op,gs,1E-4,18000,2);           // an inverse operator
  InverseOperatorResult r;

  loop.apply(x,b,r);

  // Solve system using a CG method with a Jacobi preconditioner
  SeqJac<CM_BCRS,TestVector,TestVector,2> jac(A,1,1);                // Jacobi preconditioner
  CGSolver<TestVector> cgSolver(op,jac,1E-4,18000,2);           // an inverse operator

  cgSolver.apply(x,b,r);

  // Solve system using a GMRes solver without preconditioner at all
  // Fancy (but only) way to not have a preconditioner at all
  Richardson<TestVector,TestVector> richardson(1.0);

  // Preconditioned conjugate-gradient solver
  RestartedGMResSolver<TestVector> gmres(op,
                                         richardson,
                                         1e-4,  // desired residual reduction factor
                                         5,     // number of iterations between restarts
                                         100,   // maximum number of iterations
                                         2);    // verbosity of the solver

  gmres.apply(x, b, r);

  printvector(std::cout,x[_0],"solution x1","entry",11,9,1);
  printvector(std::cout,x[_1],"solution x2","entry",11,9,1);

  return 0;
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
