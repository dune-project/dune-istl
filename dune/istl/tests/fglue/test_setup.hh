#ifndef FGLUE_TEST_SETUP_HH
#define FGLUE_TEST_SETUP_HH

namespace Test
{
  struct Base{};
  struct Derived : Base{};
  struct Other{};

  struct MakeBase
  {
    template <class...>
    struct apply
    {
      using type = Base;
    };
  };

  struct MakeDerived
  {
    template <class...>
    struct apply
    {
      using type = Derived;
    };
  };

  struct MakeOther
  {
    template <class...>
    struct apply
    {
      using type = Other;
    };
  };
}

#endif // FGLUE_TEST_SETUP_HH
