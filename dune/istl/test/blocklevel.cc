// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \file
 * \brief Test the MultiTypeBlockMatrix data structure
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/fmatrix.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/blocklevel.hh>

template<int i, int j>
using FMBlock = Dune::FieldMatrix<double,i,j>;

int main(int argc, char** argv)
{
  using namespace Dune;

  using RowType0 = MultiTypeBlockVector<Matrix<FMBlock<3,3>>, Matrix<FMBlock<3,1>>>;
  using RowType1 = MultiTypeBlockVector<Matrix<FMBlock<1,3>>, Matrix<FMBlock<3,3>>>;
  using RowType2 = MultiTypeBlockVector<Matrix<FMBlock<1,3>>, Matrix<double>>;

  {
    using MTBM = MultiTypeBlockMatrix<RowType0, RowType1, RowType1>;
    static_assert(blockLevel<MTBM>() == 3, "Wrong block level!");
  }
  {
    using MTBM = MultiTypeBlockMatrix<RowType0, RowType1, RowType2>;
    static_assert(maxBlockLevel<MTBM>() == 3, "Wrong block level!");
    static_assert(minBlockLevel<MTBM>() == 2, "Wrong block level!");
  }
  {
    static_assert(blockLevel<double>() == 0, "Wrong block level!");
    static_assert(blockLevel<FMBlock<3,1>>() == 1, "Wrong block level!");
    static_assert(blockLevel<Matrix<FMBlock<3,1>>>() == 2, "Wrong block level!");
    static_assert(blockLevel<BCRSMatrix<FMBlock<3,1>>>() == 2, "Wrong block level!");
  }

  return 0;
}
