#ifndef DUNE_ISTL_TESTS_MOCK_TRIVIAL_PRECONDITIONER_HH
#define DUNE_ISTL_TESTS_MOCK_TRIVIAL_PRECONDITIONER_HH

#include "dune/istl/preconditioner.hh"
#include "dune/istl/solvercategory.hh"

#include "vector.hh"

namespace Dune
{
  namespace Mock
  {
    class Vector;

    class TrivialPreconditioner : public Preconditioner<Vector,Vector>
    {
    public:
      static constexpr typename SolverCategory::Category category = SolverCategory::sequential;

      void pre( Vector&, Vector& ) final override;

      void apply( Vector& x, const Vector& y ) final override;

      void post( Vector& ) final override;
    };
  }
}

#endif // DUNE_ISTL_TESTS_MOCK_TRIVIAL_PRECONDITIONER_HH
