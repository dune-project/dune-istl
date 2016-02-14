#include <gtest/gtest.h>

#include "dune/istl/solvers.hh"
#include "dune/istl/residual_based_termination_criterion.hh"

#include "dune/istl/tests/mock/step.hh"

namespace
{
  struct TestResidualBasedTerminationCriterion : ::testing::Test
  {
    TestResidualBasedTerminationCriterion()
    {
      terminationCriterion.connect(step);
      terminationCriterion.reset();
    }

    Dune::KrylovTerminationCriterion::ResidualBased<double> terminationCriterion;
    Dune::Mock::Step step;
  };
}


TEST(ResidualBasedTerminationCriterion,DeathTests)
{
  auto terminationCriterion = Dune::KrylovTerminationCriterion::ResidualBased<double>();

  ASSERT_DEATH( static_cast<bool>(terminationCriterion), "" );
  ASSERT_DEATH( terminationCriterion.reset(), "" );
  ASSERT_DEATH( terminationCriterion.errorEstimate(), "" );
}

TEST_F(TestResidualBasedTerminationCriterion, Terminate)
{
  terminationCriterion.setRelativeAccuracy( 1e-3 );

  ASSERT_FALSE( static_cast<bool>(terminationCriterion) );

  step.residualNorm_ = 1e-1 * terminationCriterion.relativeAccuracy();
  ASSERT_TRUE( static_cast<bool>(terminationCriterion) );
}

TEST_F(TestResidualBasedTerminationCriterion, ErrorEstimate)
{
  ASSERT_EQ( terminationCriterion.errorEstimate(), 1. );

  auto tol = 1e-3;
  auto initialResidual = step.residualNorm();
  step.residualNorm_ = tol;

  ASSERT_FALSE( static_cast<bool>(terminationCriterion) );
  ASSERT_EQ( terminationCriterion.errorEstimate(), tol/initialResidual );
}
