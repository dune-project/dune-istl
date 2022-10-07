// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// start with including some headers
#include "config.h"

#define DISABLE_AMG_DIRECTSOLVER 1

#include <iostream>               // for input/output to shell
#include <fstream>                // for input/output to files
#include <vector>                 // STL vector class
#include <complex>

#include <cmath>                 // Yes, we do some math here
#include <sys/times.h>            // for timing measurements

#include <dune/common/alignedallocator.hh>
#include <dune/common/classname.hh>
#include <dune/common/debugalign.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#if HAVE_VC
#include <dune/common/simd/vc.hh>
#endif
#include <dune/common/timer.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/basearray.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <dune/istl/test/laplacian.hh>
#include <dune/istl/test/multirhstest.hh>

int main (int argc, char ** argv)
{
  test_all<float>();
  test_all<double>();

  test_all<Dune::AlignedNumber<double> >();

#if HAVE_VC
  test_all<Vc::float_v>();
  test_all<Vc::double_v>();
  test_all<Vc::Vector<double, Vc::VectorAbi::Scalar>>();
  test_all<Vc::SimdArray<double,2>>();
  test_all<Vc::SimdArray<double,2,Vc::Vector<double, Vc::VectorAbi::Scalar>,1>>();
  test_all<Vc::SimdArray<double,8>>();
  test_all<Vc::SimdArray<double,8,Vc::Vector<double, Vc::VectorAbi::Scalar>,1>>();
#endif

  test_all<double>(8);

  return 0;
}
