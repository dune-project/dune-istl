#include <gtest/gtest.h>

#include "dune/istl/fglue/tmp/and.hh"
#include "dune/istl/fglue/tmp/apply.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/delay.hh"
#include "dune/istl/fglue/tmp/true_false.hh"
#include "dune/istl/fglue/tmp/variadic.hh"

using namespace Dune::FGlue;

TEST(TMP, Delay)
{
  using Reference = Apply<And,True,True>;
  auto value = isTrue<Reference>();
  ASSERT_TRUE(value);

  using Delayed = Apply<Delay<And>,Constant<True>,Constant<True>>;
  value = isTrue< Apply<Delayed> >();
  ASSERT_TRUE(value);

  value = isTrue< Apply<Variadic<Delayed,And>,int,char,double> >();
  ASSERT_TRUE(value);
}
