#include <string>

#include <gtest/gtest.h>

#include "build-cmake/config.h"
#include "dune/istl/solvers.hh"

#include "dune/istl/generic_iterative_method.hh"

#include "mock/step.hh"
#include "mock/termination_criteria.hh"
#include "mock/vector.hh"

namespace Mock = Dune::Mock;

using Mock::Step;
using Mock::RestartingStep;
using Mock::TerminatingStep;
using Mock::TerminationCriterion;
using Mock::MixinTerminationCriterion;

namespace
{
  inline double testAccuracy()
  {
    return 1e-6;
  }
}


TEST(GenericIterativeMethod,NotConvergedInZeroSteps)
{
  auto iterativeMethod = Dune::makeGenericIterativeMethod(Step(),TerminationCriterion<Step>());
  iterativeMethod.setMaxSteps(0);
  Mock::Vector x, b;

  EXPECT_FALSE( iterativeMethod.getStep().wasInitialized );
  Dune::InverseOperatorResult info;
  iterativeMethod.apply(x,b,info);
  EXPECT_FALSE( info.converged );
  EXPECT_TRUE( iterativeMethod.getStep().wasInitialized );
  EXPECT_TRUE( iterativeMethod.getTerminationCriterion().wasInitialized );
  EXPECT_FALSE( iterativeMethod.getStep().wasReset );
}

TEST(GenericIterativeMethod,NotConverged)
{
  auto iterativeMethod = Dune::makeGenericIterativeMethod(Step(),TerminationCriterion<Step>(false));
  iterativeMethod.setMaxSteps(10);
  Mock::Vector x, b;

  EXPECT_FALSE( iterativeMethod.getStep().wasInitialized );
  Dune::InverseOperatorResult info;
  iterativeMethod.apply(x,b,info);
  EXPECT_FALSE( info.converged );
  EXPECT_TRUE( iterativeMethod.getStep().wasInitialized );
  EXPECT_TRUE( iterativeMethod.getTerminationCriterion().wasInitialized );
  EXPECT_FALSE( iterativeMethod.getStep().wasReset );
}

TEST(GenericIterativeMethod,Converged)
{
  auto iterativeMethod = Dune::makeGenericIterativeMethod(Step(),TerminationCriterion<Step>());
  iterativeMethod.setMaxSteps(2);
  Mock::Vector x, b;

  EXPECT_FALSE( iterativeMethod.getStep().wasInitialized );
  Dune::InverseOperatorResult info;
  iterativeMethod.apply(x,b,info);
  EXPECT_TRUE( info.converged );
  EXPECT_TRUE( iterativeMethod.getStep().wasInitialized );
  EXPECT_TRUE( iterativeMethod.getTerminationCriterion().wasInitialized );
  EXPECT_FALSE( iterativeMethod.getStep().wasReset );
}

TEST(GenericIterativeMethod,Converged_TerminatingStep)
{
  TerminatingStep step;
  step.doTerminate = true;
  auto iterativeMethod = Dune::makeGenericIterativeMethod(step,TerminationCriterion<TerminatingStep>());
  iterativeMethod.setMaxSteps(1);
  Mock::Vector x, b;

  Dune::InverseOperatorResult info;
  iterativeMethod.apply(x,b,info);
  EXPECT_TRUE( info.converged );
  EXPECT_TRUE( iterativeMethod.getStep().wasInitialized );
  EXPECT_FALSE( iterativeMethod.getStep().wasReset );
}

TEST(GenericIterativeMethod,RestartAndTerminate)
{
  RestartingStep step(true);
  auto iterativeMethod = Dune::makeGenericIterativeMethod(step,TerminationCriterion<RestartingStep>(false));
  iterativeMethod.setMaxSteps(1);
  Mock::Vector x, b;

  Dune::InverseOperatorResult info;
  iterativeMethod.apply(x,b,info);
  EXPECT_TRUE( info.converged );
  EXPECT_TRUE( iterativeMethod.getStep().wasInitialized );
  EXPECT_TRUE( iterativeMethod.getStep().wasReset );
}

TEST(GenericIterativeMethod,MixinParameters)
{
  auto iterativeMethod = Dune::makeGenericIterativeMethod(Step(),MixinTerminationCriterion<Step>());
  iterativeMethod.setAbsoluteAccuracy(testAccuracy());
  EXPECT_DOUBLE_EQ( iterativeMethod.getTerminationCriterion().absoluteAccuracy() , testAccuracy() );
  iterativeMethod.setRelativeAccuracy(testAccuracy());
  EXPECT_DOUBLE_EQ( iterativeMethod.getTerminationCriterion().relativeAccuracy() , testAccuracy() );
  iterativeMethod.setMinimalAccuracy(testAccuracy());
  EXPECT_DOUBLE_EQ( iterativeMethod.getTerminationCriterion().minimalAccuracy() , testAccuracy() );
  iterativeMethod.setEps(testAccuracy());
  EXPECT_DOUBLE_EQ( iterativeMethod.getTerminationCriterion().eps() , testAccuracy() );
  iterativeMethod.setVerbosityLevel(2);
  EXPECT_EQ( iterativeMethod.getTerminationCriterion().verbosityLevel() , 2u );
  EXPECT_TRUE( iterativeMethod.getTerminationCriterion().is_verbose() );
}
