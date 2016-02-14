#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

using namespace Dune::FGlue;

TEST(TMP,Apply)
{
  auto true_v = Apply< Constant<True> >::value;
  auto false_v = Apply< Constant<False> >::value;
  ASSERT_TRUE( true_v );
  ASSERT_FALSE( false_v );
}
