#ifndef DUNE_ISTL_TESTS_MOCK_STEP_HH
#define DUNE_ISTL_TESTS_MOCK_STEP_HH

#include "vector.hh"

#include <string>

namespace Dune
{
  namespace Mock
  {
    struct Step
    {
      using domain_type = Vector;
      using range_type = Vector;

      void init(Vector& x, Vector& b);

      void reset(Vector& x, Vector& b);

      void compute(Vector& x, Vector& b);

      Vector getFinalIterate();

      double residualNorm() const;

      double preconditionedResidualNorm() const;

      double alpha() const;

      double length() const;

      void postProcess(Vector&);

      std::string name() const;

      bool wasInitialized = false, wasReset = false;
      Vector x0, b0;
      double residualNorm_ = 1;
      double preconditionedResidualNorm_ = 1;
      double alpha_ = 1;
      double length_ = 1;
    };

    struct RestartingStep
    {
      using domain_type = Vector;
      using range_type = Vector;

      RestartingStep(bool restart = false);

      void init(Vector& x, Vector& b);

      void reset(Vector& x, Vector& b);

      void compute(Vector& x, Vector& b);

      bool terminate() const;

      bool restart() const;

      void postProcess(Vector&);

      Vector getFinalIterate();

      std::string name() const;

      bool wasInitialized = false, wasReset = false, doRestart = false;
      mutable bool doTerminate = false;
      Vector x0, b0;
    };

    struct TerminatingStep
    {
      using domain_type = Vector;
      using range_type = Vector;

      void init(Vector& x, Vector& b);

      void reset(Vector& x, Vector& b);

      void compute(Vector& x, Vector& b);

      Vector getFinalIterate();

      void postProcess(Vector&);

      std::string name() const;

      bool terminate() const;

      bool wasInitialized = false, wasReset = false, doTerminate=false;
      Vector x0, b0;
    };

  }
}

#endif // DUNE_ISTL_TESTS_MOCK_STEP_HH
