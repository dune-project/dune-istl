// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_PRECONDITIONERFACTORIES_HH
#define DUNE_ISTL_PRECONDITIONERFACTORIES_HH

#include <dune/istl/preconditioners.hh>

namespace Dune {

  namespace PreconditionerFactories {
#ifndef __cpp_inline_variables
    namespace {
#endif

      DUNE_INLINE_VARIABLE const auto richardson =
        [](auto lin_op, const ParameterTree& config, auto comm) {
          using Domain = typename std::decay_t<decltype(*lin_op)>::domain_type;
          using Range = typename std::decay_t<decltype(*lin_op)>::range_type;
          using field_type = Simd::Scalar<typename Domain::field_type>;

          field_type relaxation = config.template get<field_type>("relaxation", 1.0);

          return std::make_shared<Dune::Richardson<Domain, Range>>(relaxation);
        };


#ifndef __cpp_inline_variables
    }
#endif
  }
}


#endif
