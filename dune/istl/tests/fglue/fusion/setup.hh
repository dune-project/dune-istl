// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#ifndef FGLUE_FUSION_TEST_SETUP_HH
#define FGLUE_FUSION_TEST_SETUP_HH


struct TestData
{
  void setValue(int value)
  {
    value_ = value;
    for(auto* connection : connected_)
      connection->setValue(value_);
  }

  int getValue() const
  {
    return value_;
  }

  void attach(TestData& other)
  {
    for(auto* connection : connected_)
      if( connection == &other )
        return;

    connected_.push_back(&other);
  }

  void detach(TestData& other)
  {
    auto iend = end(connected_);
    for(auto i=begin(connected_); i!=iend; ++i)
      if( *i == &other )
        connected_.erase(i);
  }

private:
  int value_ = 0;
  std::vector<TestData*> connected_;
};

struct A : TestData{};
struct B{};

#endif // FGLUE_FUSION_TEST_SETUP_HH
