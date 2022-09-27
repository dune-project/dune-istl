// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \file
 * \brief Test block level function
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/blocklevel.hh>

template<int i, int j>
using FMBlock = Dune::FieldMatrix<double,i,j>;

template<int i>
using FVBlock = Dune::FieldVector<double,i>;

int main(int argc, char** argv)
{
  using namespace Dune;

  static_assert(blockLevel<double>() == 0, "Wrong block level!");

  // vector tests
  static_assert(blockLevel<FVBlock<3>>() == 1, "Wrong block level!");
  static_assert(blockLevel<BlockVector<FVBlock<3>>>() == 2, "Wrong block level!");

  using BlockType0 = BlockVector<FVBlock<3>>;
  using BlockType1 = BlockVector<double>;
  using MTBV0 = MultiTypeBlockVector<BlockType0, BlockType0>;
  static_assert(blockLevel<MTBV0>() == 3, "Wrong block level!");

  using MTBV1 = MultiTypeBlockVector<BlockType0, BlockType1>;
  static_assert(maxBlockLevel<MTBV1>() == 3, "Wrong block level!");
  static_assert(minBlockLevel<MTBV1>() == 2, "Wrong block level!");
  static_assert(!hasUniqueBlockLevel<MTBV1>(), "Block level shouldn't be unique!");

  // matrix tests
  static_assert(blockLevel<FMBlock<3,1>>() == 1, "Wrong block level!");
  static_assert(blockLevel<Matrix<FMBlock<3,1>>>() == 2, "Wrong block level!");
  static_assert(blockLevel<BCRSMatrix<FMBlock<3,1>>>() == 2, "Wrong block level!");

  using RowType0 = MultiTypeBlockVector<Matrix<FMBlock<3,3>>, Matrix<FMBlock<3,1>>>;
  using RowType1 = MultiTypeBlockVector<Matrix<FMBlock<1,3>>, Matrix<FMBlock<3,3>>>;
  using RowType2 = MultiTypeBlockVector<Matrix<FMBlock<1,3>>, Matrix<double>>;
  using MTBM0 = MultiTypeBlockMatrix<RowType0, RowType1, RowType1>;
  static_assert(blockLevel<MTBM0>() == 3, "Wrong block level!");

  using MTBM1 = MultiTypeBlockMatrix<RowType0, RowType1, RowType2>;
  static_assert(maxBlockLevel<MTBM1>() == 3, "Wrong block level!");
  static_assert(minBlockLevel<MTBM1>() == 2, "Wrong block level!");
  static_assert(!hasUniqueBlockLevel<MTBM1>(), "Block level shouldn't be unique!");

  return 0;
}
