#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/and.hh"
#include "dune/istl/fglue/tmp/create_missing_base_classes.hh"

using namespace Dune::FGlue;

struct A{};
struct B{};
struct C{};

struct V{};

struct X : A{};
struct Y : B{};
struct Z : C{};


TEST(TMP,DetectBaseClassCandidates)
{
  using BaseClassIdentifier = IsBaseOfOneOf<X,Y,Z>;

  auto value = isTrue< Apply<BaseClassIdentifier,A> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<BaseClassIdentifier,B> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<BaseClassIdentifier,C> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<BaseClassIdentifier,Z> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<BaseClassIdentifier,V> >();
  ASSERT_FALSE(value);


  using StorageOperation = StoreIf<IsBaseOfOneOf<X,Y,Z>>;
  using BaseClasses = Apply< Variadic< StorageOperation , Compose> , A , B , C , V>;

  value = std::is_base_of<A,BaseClasses>::value;
  ASSERT_TRUE(value);

  value = std::is_base_of<B,BaseClasses>::value;
  ASSERT_TRUE(value);

  value = std::is_base_of<C,BaseClasses>::value;
  ASSERT_TRUE(value);

  value = std::is_base_of<V,BaseClasses>::value;
  ASSERT_FALSE(value);


//  using CheckedBaseClasses = CreateMissingBases<A,B,C,V>::BaseOf<X,Y,Z>::NotBaseOf<X>;

//  value = std::is_base_of<A,CheckedBaseClasses>::value;
//  ASSERT_FALSE(value);

//  value = std::is_base_of<B,CheckedBaseClasses>::value;
//  ASSERT_TRUE(value);

//  value = std::is_base_of<C,CheckedBaseClasses>::value;
//  ASSERT_TRUE(value);

//  value = std::is_base_of<V,CheckedBaseClasses>::value;
//  ASSERT_FALSE(value);
}
