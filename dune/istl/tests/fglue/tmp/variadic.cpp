#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/and.hh"
#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/delay.hh"
#include "dune/istl/fglue/tmp/is_derived_from.hh"
#include "dune/istl/fglue/tmp/make.hh"
#include "dune/istl/fglue/tmp/or.hh"
#include "dune/istl/fglue/tmp/true_false.hh"
#include "dune/istl/fglue/tmp/variadic.hh"
#include "dune/istl/tests/fglue/test_setup.hh"

using namespace Dune::FGlue;
using namespace Test;

TEST(TMP,Variadic_Logic_Apply)
{
  auto value = isTrue< Apply<Binary2Variadic<And>,True,True> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<Binary2Variadic<And>,True,True,True> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<Binary2Variadic<And>,True,True,False> >();
  ASSERT_FALSE(value);

  value = isTrue< Apply<Binary2Variadic<And>,True,False,True> >();
  ASSERT_FALSE(value);

  value = isTrue< Apply<Binary2Variadic<And>,False,True,True> >();
  ASSERT_FALSE(value);
}

TEST(TMP,Make)
{
  using Operation = Apply< Make<IsDerivedFrom> , Base >;
  auto value = std::is_same< Operation , IsDerivedFrom<Base> >::value;
  ASSERT_TRUE(value);
}

TEST(TMP,VariadicMakeFromTemplate)
{
  using Generator = Variadic< Make<IsDerivedFrom> , Delay<And> >;

  using Operation_BaseBase = Apply< Generator , Base , Base >;
  auto isBaseDerived = isTrue< Apply< Operation_BaseBase , Base > >();
  auto isDerivedDerived = isTrue< Apply< Operation_BaseBase , Derived > >();
  ASSERT_TRUE(isBaseDerived);
  ASSERT_TRUE(isDerivedDerived);

  using Operation_DerivedDerived = Apply< Generator , Derived , Derived >;
  isBaseDerived = isTrue< Apply< Operation_DerivedDerived , Base > >();
  isDerivedDerived = isTrue< Apply< Operation_DerivedDerived , Derived > >();
  ASSERT_FALSE(isBaseDerived);
  ASSERT_TRUE(isDerivedDerived);

  using Operation_BaseOther = Apply< Generator , Base , Other >;
  isBaseDerived = isTrue< Apply< Operation_BaseOther , Base > >();
  isDerivedDerived = isTrue< Apply< Operation_BaseOther , Derived > >();
  ASSERT_FALSE(isBaseDerived);
  ASSERT_FALSE(isDerivedDerived);
}

TEST(TMP, Variadic_Generate_Logic_Apply)
{
  using Generator = Variadic< Make<IsDerivedFrom> , Delay<And> >;
  using Operation = Apply< Generator , Base , Base , Base >;
  auto value = isTrue< Apply< Variadic<Operation,And>, Base , Derived , Base > >();
  ASSERT_TRUE(value);

  using Operation2 = Apply< Generator , Base , Derived , Base >;
  value = isTrue< Apply< Variadic<Operation2,And>, Base , Derived , Base > >();
  ASSERT_FALSE(value);
}
