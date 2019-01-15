// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/** \file \brief Test the preconditioners in the file `preconditioners.hh`
 */

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/test/laplacian.hh>

using namespace Dune;

template <class Matrix, class Vector>
void setupProblem(Matrix& matrix, Vector& b)
{
  int n=100;

  setupLaplacian(matrix,n);

  b.resize(n*n);
  Vector x(n*n);

  // Construct right-hand side such that '1' is the solution
  b=0;
  x=1;
  matrix.mv(x, b);
}

template <class Matrix, class Vector, class Preconditioner>
void testPreconditioner(const Matrix& matrix, const Vector& b, Vector& x, Preconditioner& prec)
{
  using Operator = MatrixAdapter<Matrix,Vector,Vector>;

  Operator linearOperator(matrix);
  InverseOperatorResult result;

  Dune::LoopSolver<Vector> solver(linearOperator, prec, 1e-8,15,2);
  auto residual = b;
  solver.apply(x,residual,result);
}

template <class Matrix, class Vector>
void testAllPreconditioners(const Matrix& matrix, const Vector& b)
{
  Vector x = b;  // Set the correct size
  x = 0;
  SeqSSOR<Matrix,Vector,Vector> seqSSOR(matrix, 1,1.0);
  testPreconditioner(matrix, b, x, seqSSOR);

  x = 0;
  SeqSOR<Matrix,Vector,Vector> seqSOR(matrix, 1,1.0);
  testPreconditioner(matrix, b, x, seqSOR);

  x = 0;
  SeqGS<Matrix,Vector,Vector> seqGS(matrix, 1,1.0);
  testPreconditioner(matrix, b, x, seqGS);

  x = 0;
  SeqJac<Matrix,Vector,Vector> seqJac(matrix, 1,1.0);
  testPreconditioner(matrix, b, x, seqJac);

  x = 0;
  SeqILU<Matrix,Vector,Vector> seqILU(matrix, 3, 1.2, true);
  testPreconditioner(matrix, b, x, seqILU);

  x = 0;
  Richardson<Vector,Vector> richardson(1.5);
  testPreconditioner(matrix, b, x, richardson);

  x = 0;
  SeqILDL<Matrix,Vector,Vector> seqILDL(matrix, 1.2);
  testPreconditioner(matrix, b, x, seqILDL);
}

int main() try
{
  {
    using Matrix = BCRSMatrix<FieldMatrix<double,1,1> >;
    using Vector = BlockVector<FieldVector<double,1> >;
    Matrix matrix;
    Vector b,x;
    setupProblem(matrix, b);

    testAllPreconditioners(matrix, b);
  }

  return 0;
}
catch (std::exception& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
