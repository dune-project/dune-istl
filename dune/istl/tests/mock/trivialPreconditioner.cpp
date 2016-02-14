#include "trivialPreconditioner.hh"

namespace Dune
{
  namespace Mock
  {
    void TrivialPreconditioner::pre(Vector &, Vector &){}

    void TrivialPreconditioner::apply( Vector& x, const Vector& y )
    {
      x = y;
    }

    void TrivialPreconditioner::post(Vector &){}
  }
}
