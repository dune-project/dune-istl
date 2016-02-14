#ifndef DUNE_ISTL_TESTS_MOCK_TERMINATION_CRITERIA_HH
#define DUNE_ISTL_TESTS_MOCK_TERMINATION_CRITERIA_HH

#include "dune/istl/mixins/absoluteAccuracy.hh"
#include "dune/istl/mixins/minimalAccuracy.hh"
#include "dune/istl/mixins/relativeAccuracy.hh"
#include "dune/istl/mixins/eps.hh"
#include "dune/istl/mixins/verbosity.hh"

#include "dune/istl/optional.hh"

namespace Dune
{
  namespace Mock
  {
    template <class Step>
    struct TerminationCriterion : Dune::Mixin::RelativeAccuracy<double>
    {
      TerminationCriterion(bool val = true)
        : value(val)
      {}

      void reset()
      {
        wasInitialized = true;
      }

      operator bool() const
      {
        if( step_ != nullptr && Optional::terminate(*step_))
          return true;

        return value;
      }

      void connect(const Step& step)
      {
        step_ = &step;
      }

      double absoluteError() const
      {
        return 1;
      }

      double errorEstimate() const
      {
        return 1;
      }

      template <class InverseOperatorResult>
      void write(InverseOperatorResult&) const
      {}

      bool wasInitialized = false;
      bool value = true;
      const Step* step_ = nullptr;
    };

    template <class Step>
    struct MixinTerminationCriterion
        : Dune::Mixin::AbsoluteAccuracy<double>,
          Dune::Mixin::RelativeAccuracy<double>,
          Dune::Mixin::MinimalAccuracy<double>,
          Dune::Mixin::Eps<double>,
          Dune::Mixin::Verbosity
    {
      void reset()
      {
        wasInitialized = true;
      }

      operator bool() const
      {
        return true;
      }

      void connect(const Step&)
      {}

      double absoluteError() const
      {
        return 1;
      }

      double errorEstimate() const
      {
        return 1;
      }

      template <class InverseOperatorResult>
      void write(InverseOperatorResult& res) const
      {}

      bool wasInitialized = false;
    };
  }
}

#endif // DUNE_ISTL_TESTS_MOCK_TERMINATION_CRITERIA_HH
