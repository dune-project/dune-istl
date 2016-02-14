#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/is_derived_from.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;

TEST(TMP,IsDerivedFrom)
{
  auto baseBase =  isTrue<Apply<IsDerivedFrom<Base>,Base>>();
  auto baseDerived = isTrue<Apply<IsDerivedFrom<Base>,Derived>>();
  auto derivedBase = isTrue<Apply<IsDerivedFrom<Derived>,Base>>();

  ASSERT_TRUE( baseBase );
  ASSERT_TRUE( baseDerived );
  ASSERT_FALSE( derivedBase );
}

TEST(TMP,IsNotDerivedFrom)
{
  auto baseBase =  isTrue<Apply<IsNotDerivedFrom<Base>,Base>>();
  auto baseDerived = isTrue<Apply<IsNotDerivedFrom<Base>,Derived>>();
  auto derivedBase = isTrue<Apply<IsNotDerivedFrom<Derived>,Base>>();

  ASSERT_FALSE( baseBase );
  ASSERT_FALSE( baseDerived );
  ASSERT_TRUE( derivedBase );
}
