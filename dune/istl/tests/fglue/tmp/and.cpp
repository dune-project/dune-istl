#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/and.hh"
#include "dune/istl/fglue/tmp/apply.hh"

using namespace Dune::FGlue;

TEST(TMP,And)
{
  auto trueTrue = isTrue< Apply<And,True,True> >();
  auto trueFalse = isTrue< Apply<And,True,False> >();
  auto falseTrue = isTrue< Apply<And,False,True> >();
  auto falseFalse = isTrue< Apply<And,False,False> >();

  ASSERT_TRUE( trueTrue );
  ASSERT_FALSE( trueFalse );
  ASSERT_FALSE( falseTrue );
  ASSERT_FALSE( falseFalse );
}
