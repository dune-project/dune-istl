#include "step.hh"

namespace Dune
{
  namespace Mock
  {
    void Step::init(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasInitialized = true;
    }

    void Step::reset(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasReset = true;
    }

    void Step::compute(Vector& x, Vector& b)
    {
      x0 += b0;
    }

    Vector Step::getFinalIterate()
    {
      return x0;
    }

    double Step::residualNorm() const
    {
      return residualNorm_;
    }

    double Step::preconditionedResidualNorm() const
    {
      return preconditionedResidualNorm_;
    }

    double Step::alpha() const
    {
      return alpha_;
    }

    double Step::length() const
    {
      return length_;
    }

    void Step::postProcess(Vector&)
    {}

    std::string Step::name() const
    {
      return "dummy step";
    }


    RestartingStep::RestartingStep(bool restart)
      : doRestart(restart)
    {}

    void RestartingStep::init(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasInitialized = true;
    }

    void RestartingStep::reset(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasReset = true;
    }

    void RestartingStep::compute(Vector& x, Vector& b)
    {
      x0 += b0;
    }

    bool RestartingStep::terminate() const
    {
      return doTerminate && wasReset;
    }

    bool RestartingStep::restart() const
    {
      doTerminate = true;
      return doRestart;
    }

    Vector RestartingStep::getFinalIterate()
    {
      return x0;
    }

    void RestartingStep::postProcess(Vector&)
    {}

    std::string RestartingStep::name() const
    {
      return "dummy step";
    }


    void TerminatingStep::init(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasInitialized = true;
    }

    void TerminatingStep::reset(Vector& x, Vector& b)
    {
      x0 = x;
      b0 = b;
      wasReset = true;
    }

    void TerminatingStep::compute(Vector& x, Vector& b)
    {
      x0 += b0;
    }

    Vector TerminatingStep::getFinalIterate()
    {
      return x0;
    }

    void TerminatingStep::postProcess(Vector&)
    {}

    std::string TerminatingStep::name() const
    {
      return "dummy step";
    }

    bool TerminatingStep::terminate() const
    {
      return doTerminate;
    }
  }
}
