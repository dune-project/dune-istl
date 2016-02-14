#include <cmath>
#include <iterator>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/timer.hh>
#include "dune/istl/bvector.hh"
#include "dune/istl/operators.hh"
#include "dune/istl/preconditioners.hh"
#include "dune/istl/trcg_solver.hh"
#include "laplacian.hh"
#include "setup.hh"

namespace
{
  using namespace TestSetup;

  using MatrixBlock = Dune::FieldMatrix<double,blockSize,blockSize>;
  using BCRSMat = Dune::BCRSMatrix<MatrixBlock>;
  using VectorBlock = Dune::FieldVector<double,blockSize>;
  using BVector = Dune::BlockVector<VectorBlock>;
  using Operator = Dune::MatrixAdapter<BCRSMat,BVector,BVector>;
  using Preconditioner = Dune::SeqJac<BCRSMat,BVector,BVector>;
  using Solver = Dune::TCGSolver<BVector>;


  struct TestTRCG_EnergyErrorTermination : ::testing::Test
  {
    TestTRCG_EnergyErrorTermination()
      : mat(),
        A( mat )
    {}

    void SetUp()
    {
      setupLaplacian(mat,N);
      P = std::unique_ptr<Preconditioner>( new Preconditioner(mat, 1, 1.0) );
      cg = std::unique_ptr<Solver>( new Solver(A, *P, tol, maxSteps, verbosityLevel) );
    }

    BCRSMat mat;
    Operator A;
    std::unique_ptr<Preconditioner> P;
    std::unique_ptr<Solver> cg;
  };
}

TEST_F(TestTRCG_EnergyErrorTermination, Laplacian)
{
  BVector b(N*N), x(N*N);

  x=1;
  mat.mv(x, b);
  x=0;

  Dune::InverseOperatorResult res;
  cg->apply(x, b, res);

  TestSetup::compareWithStoredSolution(x, TestSetup::tol);

  ASSERT_TRUE( res.converged );
  ASSERT_EQ( res.iterations, TestSetup::expectedIterations_EnergyError );
}
