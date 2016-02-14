#ifndef DUNE_ISTL_FUNCTIONAL_TEST_SETUP_HH
#define DUNE_ISTL_FUNCTIONAL_TEST_SETUP_HH

#include <fstream>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

namespace TestSetup
{
  constexpr int blockSize=1;

  constexpr int N=100;

  constexpr unsigned maxSteps = 1000;

  constexpr unsigned verbosityLevel = 0;

  constexpr double tol = 1e-3;

  constexpr double condition = 5;

  constexpr unsigned expectedIterations_ResidualBasedError = 116;

  constexpr unsigned expectedIterations_EnergyError = 130;


  inline std::string toScientificString(double tol)
  {
    auto counter = 0u;
    while( tol < 1. )
    {
      ++counter;
      tol *= 10;
    }

    return std::to_string( static_cast<int>(tol) ) + "e-" + std::to_string(counter);
  }

  inline std::string referenceFileName()
  {
    return "../functional_tests/solution_laplacian_N_100_tol_1e-15";
  }

  template <class Vector>
  void compareWithStoredSolution(const Vector& x, double tolerance)
  {
    using Index = decltype( x.size() );

    std::ifstream file;
    file.open( referenceFileName() );
    assert( file.is_open() );

    double reference = 0;
    for( Index i=0; i<x.size(); ++i )
    {
      file >> reference;
      ASSERT_NEAR( x[i][0], reference, tolerance );
    }
  }
}

#endif // DUNE_ISTL_FUNCTIONAL_TEST_SETUP_HH
