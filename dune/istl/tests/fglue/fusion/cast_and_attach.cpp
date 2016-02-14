#include <gtest/gtest.h>

#include "dune/istl/fglue/fusion/cast_and_attach.hh"
#include "dune/istl/fglue/fusion/apply_if.hh"
#include "dune/istl/fglue/tmp/constant.hh"
#include "dune/istl/fglue/tmp/true_false.hh"

#include "setup.hh"

using namespace Dune::FGlue;
using namespace Dune::FGlue::Fusion;

TEST(Fusion,CastAndAttach)
{
  A a, c;
  CastAndAttach<TestData> castAndAttach(c);

  castAndAttach(a);

  EXPECT_EQ( a.getValue() , 0 );
  c.setValue(2);
  EXPECT_EQ( a.getValue() , 2 );
}

TEST(Fusion,CastAndDetach)
{
  A a, c;
  CastAndAttach<TestData> castAndAttach(c);
  CastAndDetach<TestData> castAndDetach(c);

  castAndAttach(a);
  castAndAttach(a);

  EXPECT_EQ( a.getValue() , 0 );
  c.setValue(2);
  EXPECT_EQ( a.getValue() , 2 );

  castAndDetach(a);
  c.setValue(3);
  EXPECT_EQ( c.getValue() , 3 );
  EXPECT_EQ( a.getValue() , 2 );
}

TEST(Fusion,UnaryApplyIf)
{
  A a, c_true, c_false;
  UnaryApplyIf< CastAndAttach<TestData> , Constant<True> > castAndAttach_true(c_true);
  UnaryApplyIf< CastAndAttach<TestData> , Constant<False> > castAndAttach_false(c_false);

  castAndAttach_true(a);
  castAndAttach_false(a);

  EXPECT_EQ( a.getValue() , 0 );
  c_true.setValue(2);
  EXPECT_EQ( a.getValue() , 2 );
  c_false.setValue(3);
  EXPECT_EQ( a.getValue() , 2 );
}
