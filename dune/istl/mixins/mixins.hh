#ifndef DUNE_ISTL_MIXINS_HH
#define DUNE_ISTL_MIXINS_HH

/**
 * @defgroup MixinGroup
 * @ingroup ISTL_Solvers
 * @brief Contains small independent components that are frequently used (such as common algorithmic parameters) and can be added via (multiple) inheritance.
 */

#include "absoluteAccuracy.hh"
#include "eps.hh"
#include "iterativeRefinements.hh"
#include "maxSteps.hh"
#include "minimalAccuracy.hh"
#include "relativeAccuracy.hh"
#include "verbosity.hh"

#define DUNE_ISTL_MIXINS(Real) ::Dune::Mixin::AbsoluteAccuracy<Real>, ::Dune::Mixin::MinimalAccuracy<Real>, ::Dune::Mixin::RelativeAccuracy<Real>, \
  ::Dune::Mixin::Verbosity, ::Dune::Mixin::Eps<Real>, ::Dune::Mixin::IterativeRefinements, ::Dune::Mixin::MaxSteps

#endif // DUNE_ISTL_MIXINS_HH
