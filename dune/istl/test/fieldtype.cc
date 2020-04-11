// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \file
 * \brief Test field type trait
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include <type_traits>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/fieldtype.hh>

int main(int argc, char** argv)
{
  using namespace Dune;

  static_assert(std::is_same_v<FieldType<double>, double>, "Wrong field type!");

  // vector tests
  static_assert(std::is_same_v<FieldType<FieldVector<double, 3>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<FieldVector<float, 3>>, float>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<BlockVector<FieldVector<double, 3>>>, double>, "Wrong field type!");

  using BlockType0 = BlockVector<FieldVector<double, 3>>;
  using BlockType1 = BlockVector<FieldVector<float, 3>>;
  using BlockType2 = BlockVector<double>;
  using BlockType3 = BlockVector<float>;
  static_assert(std::is_same_v<FieldType<MultiTypeBlockVector<BlockType0, BlockType2>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<MultiTypeBlockVector<BlockType0, BlockType3>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<MultiTypeBlockVector<BlockType1, BlockType3>>, float>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<MultiTypeBlockVector<BlockType2, BlockType3>>, double>, "Wrong field type!");

  // matrix tests
  static_assert(std::is_same_v<FieldType<Dune::FieldMatrix<double,3,3>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<Dune::FieldMatrix<float,3,3>>, float>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<Matrix<Dune::FieldMatrix<float,3,3>>>, float>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<BCRSMatrix<Dune::FieldMatrix<double,3,3>>>, double>, "Wrong field type!");

  using DoubleBlock = MultiTypeBlockVector<BlockType0, BlockType2>;
  using FloatBlock = MultiTypeBlockVector<BlockType1, BlockType3>;
  static_assert(std::is_same_v<FieldType<MultiTypeBlockMatrix<DoubleBlock, DoubleBlock>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<MultiTypeBlockMatrix<DoubleBlock, FloatBlock>>, double>, "Wrong field type!");
  static_assert(std::is_same_v<FieldType<MultiTypeBlockMatrix<FloatBlock, FloatBlock>>, float>, "Wrong field type!");

  return 0;
}
