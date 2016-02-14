#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/bind.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/is_base_of.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;

TEST(TMP,Bind)
{
  using BoundBase = Apply< Bind , IsBaseOf<Derived > , Base >;
  auto isBaseOfDerived = isTrue< Apply< BoundBase > >();
  ASSERT_TRUE( isBaseOfDerived );

  using BoundDerived = Apply< Bind , IsBaseOf<Base> , Derived >;
  auto isBaseOfBase = isTrue< Apply< BoundDerived > >();
  ASSERT_FALSE( isBaseOfBase );
}
