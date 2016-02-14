#include <gtest/gtest.h>

#include <dune/common/typetraits.hh>

#include "dune/istl/fglue/tmp/combiner.hh"
#include "dune/istl/fglue/tmp/true_false.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;
using Dune::Empty;

TEST(TMP,DefaultCombiner)
{
  using CombinedType = Apply<DefaultCombiner,True,False>;
  auto isEmpty = std::is_same<Empty,CombinedType>::value;
  ASSERT_TRUE( isEmpty );
}

TEST(TMP,Composer)
{
  using StoreBaseBase = Apply< Compose , Base , Base >;
  auto derivedFromDerived = std::is_base_of<Derived,StoreBaseBase>::value;
  auto derivedFromBase = std::is_base_of<Base,StoreBaseBase>::value;
  auto derivedFromEmpty = std::is_base_of<Empty,StoreBaseBase>::value;
  ASSERT_FALSE( derivedFromDerived );
  ASSERT_TRUE( derivedFromBase );
  ASSERT_FALSE( derivedFromEmpty );

  using StoreBaseDerived = Apply< Compose , Base , Derived >;
  derivedFromDerived = std::is_base_of<Derived,StoreBaseDerived>::value;
  derivedFromBase = std::is_base_of<Base,StoreBaseDerived>::value;
  derivedFromEmpty = std::is_base_of<Empty,StoreBaseDerived>::value;
  ASSERT_TRUE( derivedFromDerived );
  ASSERT_TRUE( derivedFromBase );
  ASSERT_FALSE( derivedFromEmpty );

  using StoreDerivedDerived = Apply< Compose , Derived , Derived >;
  derivedFromDerived = std::is_base_of<Derived,StoreDerivedDerived>::value;
  derivedFromBase = std::is_base_of<Base,StoreDerivedDerived>::value;
  derivedFromEmpty = std::is_base_of<Empty,StoreDerivedDerived>::value;
  ASSERT_TRUE( derivedFromDerived );
  ASSERT_TRUE( derivedFromBase );
  ASSERT_FALSE( derivedFromEmpty );

  using StoreOtherOther = Apply< Compose , Empty , Empty >;
  derivedFromDerived = std::is_base_of<Derived,StoreOtherOther>::value;
  derivedFromBase = std::is_base_of<Base,StoreOtherOther>::value;
  derivedFromEmpty = std::is_base_of<Empty,StoreOtherOther>::value;
  ASSERT_FALSE( derivedFromDerived );
  ASSERT_FALSE( derivedFromBase );
  ASSERT_TRUE( derivedFromEmpty );
}
