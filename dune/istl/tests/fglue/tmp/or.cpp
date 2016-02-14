#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/or.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

using namespace Dune::FGlue;

TEST(TMP,Or)
{
  auto trueTrue = isTrue< Apply<Or,True,True> >();
  auto trueFalse = isTrue< Apply<Or,True,False> >();
  auto falseTrue = isTrue< Apply<Or,False,True> >();
  auto falseFalse = isTrue< Apply<Or,False,False> >();

  ASSERT_TRUE( trueTrue );
  ASSERT_TRUE( trueFalse );
  ASSERT_TRUE( falseTrue );
  ASSERT_FALSE( falseFalse );
}
