// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include "config.h"

#include <iostream>
#include <limits>
#include <string>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvers.hh>

template<class Solver, class Vector>
void checkSolverAbort(int &status, const std::string &name,
                      Solver &solver, Vector &x, Vector &b)
{
  try {
    Dune::InverseOperatorResult res;
    solver.apply(x, b, res);
    std::cout << "Error: " << name << "::apply() did not abort, which is "
              << "unexpected.\n"
              << "converged  = " << std::boolalpha << res.converged << "\n"
              << "iterations = " << res.iterations << "\n"
              << "reduction  = " << res.reduction  << "\n"
              << "conv_rate  = " << res.conv_rate  << "\n"
              << "elapsed    = " << res.elapsed    << "\n"
              << "solution   = {" << x << "}" << std::endl;
    status = 1; // FAIL
  }
  catch(const Dune::SolverAbort &e)
  {
    std::cout << name << "::apply() aborted, as expected" << std::endl
              << "Abort message was: " << e << std::endl;
    if(status == 77) status = 0; // PASS
  }
  catch(const std::exception &e)
  {
    std::cout << "Error: " << name << "::apply() aborted with an exception "
              << "not derived from Dune::SolverAbort, which is unexpected.\n"
              << "e.what(): " << e.what() << std::endl;
    status = 1; // FAIL
  }
  catch(...)
  {
    std::cout << "Error: " << name << "::apply() aborted with an exception "
              << "not derived from Dune::SolverAbort, which is unexpected.\n"
              << "In addition, the exception is not derived from "
              << "std::exception, so there is no further information, sorry."
              << std::endl;
    status = 1; // FAIL
  }
}

int main()
{

  int status = 77;

  // How verbose the solvers should be.  Use 2 (maximum verbosity) by default,
  // this will include all information in the logs, and for the casual user of
  // the unit tests ctest will hide the output anyway.
  int verbose = 2;

  { // CGSolver
    std::cout << "Checking CGSolver with an unsolvable system...\n"
              << "Expecting SolverAbort with a NaN defect" << std::endl;

    using Matrix = Dune::FieldMatrix<double, 2, 2>;
    using Vector = Dune::FieldVector<double, 2>;

    Matrix matrix = { { 1, 1 },
                      { 1, 1 } };
    Vector b = { 1, 2 };
    Vector x = { 0, 0 };

    Dune::MatrixAdapter<Matrix, Vector, Vector> op(matrix);
    Dune::Richardson<Vector, Vector> richardson;
    Dune::CGSolver<Vector> solver(op, richardson, 1e-10, 5000, verbose);

    checkSolverAbort(status, "CGSolver", solver, x, b);
  }

  { // BiCGSTABSolver
    std::cout << "Checking BiCGSTABSolver with an unsolvable system...\n"
              << "Expecting abs(h) < EPSILON" << std::endl;

    using Matrix = Dune::FieldMatrix<double, 2, 2>;
    using Vector = Dune::FieldVector<double, 2>;

    Matrix matrix = { { 1, 1 },
                      { 1, 1 } };
    Vector b = { 1, 2 };
    Vector x = { 0, 0 };

    Dune::MatrixAdapter<Matrix, Vector, Vector> op(matrix);
    Dune::Richardson<Vector, Vector> richardson;
    Dune::BiCGSTABSolver<Vector> solver(op, richardson, 1e-10, 5000, verbose);

    checkSolverAbort(status, "BiCGSTABSolver", solver, x, b);
  }

  // TODO:
  // - trigger "breakdown in BiCGSTAB - rho"
  // - trigger "breakdown in BiCGSTAB - omega"
  // - trigger "breakdown in GMRes"

  return status;
}
