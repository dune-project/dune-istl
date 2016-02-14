#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/is_base_of.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;

TEST(TMP,IsBaseOf)
{
  auto baseBase = isTrue< Apply<IsBaseOf<Base>,Base > >();
  auto baseDerived = isTrue< Apply<IsBaseOf<Base>,Derived> >();
  auto derivedBase = isTrue< Apply<IsBaseOf<Derived>,Base> >();

  ASSERT_TRUE( baseBase );
  ASSERT_FALSE( baseDerived );
  ASSERT_TRUE( derivedBase );
}

TEST(TMP,IsNotBaseOf)
{
auto baseBase = isTrue< Apply<IsNotBaseOf<Base >,Base > >();
auto baseDerived = isTrue< Apply<IsNotBaseOf<Base>,Derived> >();
auto derivedBase = isTrue< Apply<IsNotBaseOf<Derived>,Base> >();

ASSERT_FALSE( baseBase );
ASSERT_TRUE( baseDerived );
ASSERT_FALSE( derivedBase );
}
