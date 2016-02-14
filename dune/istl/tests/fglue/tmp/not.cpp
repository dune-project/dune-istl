#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/not.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

using namespace Dune::FGlue;

TEST(TMP,Not)
{
  auto true_v = isTrue< Apply<Not,False> >();
  auto false_v = isTrue< Apply<Not,True> >();
  ASSERT_TRUE( true_v );
  ASSERT_FALSE( false_v );
}
