#include <gtest/gtest.h>

#include <type_traits>

#include <dune/common/typetraits.hh>

#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/true_false.hh"
#include "dune/istl/fglue/tmp/store_if.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;
using Dune::Empty;

TEST(TMP,StoreIf)
{
  auto isBase = std::is_same<Base,Apply<StoreIf< Constant<True> > , Base > >::value;
  ASSERT_TRUE( isBase );

  isBase = std::is_same<Base,Apply< StoreIf< Constant<False> > , Base > >::value;
  auto isEmpty = std::is_same<Empty,Apply< StoreIf< Constant<False> > , Base > >::value;
  ASSERT_FALSE( isBase );
  ASSERT_TRUE( isEmpty );
}

TEST(TMP, StoreIfDerivedFrom)
{
  using StoredTypeBaseDerived = Apply< StoreIf< IsDerivedFrom<Base> > , Derived >;
  auto isDerived = std::is_same< Derived , StoredTypeBaseDerived >::value;
  ASSERT_TRUE( isDerived );

  using StoredTypeDerivedBase = Apply< StoreIf< IsDerivedFrom<Derived> > , Base >;
  auto isBase = std::is_same< Base , StoredTypeDerivedBase >::value;
  auto isEmpty = std::is_same< Empty , StoredTypeDerivedBase >::value;
  ASSERT_FALSE( isBase );
  ASSERT_TRUE( isEmpty );
}

TEST(TMP, StoreIfNotDerivedFrom)
{
  using StoredTypeBaseDerived = Apply< StoreIf< IsNotDerivedFrom<Base> > , Derived >;
  auto isDerived = std::is_same< Derived , StoredTypeBaseDerived >::value;
  auto isEmpty = std::is_same< Empty , StoredTypeBaseDerived >::value;
  ASSERT_FALSE( isDerived );
  ASSERT_TRUE( isEmpty );

  using StoredTypeDerivedBase = Apply< StoreIf< IsNotDerivedFrom<Derived> > , Base >;
  auto isBase = std::is_same< Base , StoredTypeDerivedBase >::value;
  ASSERT_TRUE( isBase );
}

TEST(TMP, StoreIfBaseOf)
{
  using StoredTypeBaseDerived = Apply< StoreIf< IsBaseOf<Base> > , Derived >;
  auto isDerived = std::is_same< Derived , StoredTypeBaseDerived >::value;
  auto isEmpty = std::is_same< Empty , StoredTypeBaseDerived >::value;
  ASSERT_FALSE( isDerived );
  ASSERT_TRUE( isEmpty );

  using StoredTypeDerivedBase = Apply< StoreIf< IsBaseOf<Derived> > , Base >;
  auto isBase = std::is_same< Base , StoredTypeDerivedBase >::value;
  ASSERT_TRUE( isBase );
}

TEST(TMP, StoreIfNotBaseOf)
{
  using StoredTypeBaseDerived = Apply< StoreIf< IsNotBaseOf<Base> > , Derived >;
  auto isDerived = std::is_same< Derived , StoredTypeBaseDerived >::value;
  ASSERT_TRUE( isDerived );

  using StoredTypeDerivedBase = Apply< StoreIf< IsNotBaseOf<Derived> > , Base >;
  auto isBase = std::is_same< Base , StoredTypeDerivedBase >::value;
  auto isEmpty = std::is_same< Empty , StoredTypeDerivedBase >::value;
  ASSERT_FALSE( isBase );
  ASSERT_TRUE( isEmpty );
}
