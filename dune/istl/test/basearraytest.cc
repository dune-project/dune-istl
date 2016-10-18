// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/istl/basearray.hh>

using namespace Dune;

int main()
{
  Imp::base_array<double> v1(10);
  Imp::base_array<double> v2 = v1;

  // Test constructor from base_array_unmanaged
  Imp::base_array<double> v3 = *(static_cast<Imp::base_array_unmanaged<double>*>(&v1));

  v1.resize(20);

  v1 = v2;

}
