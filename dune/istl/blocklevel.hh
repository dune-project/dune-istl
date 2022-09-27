// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKLEVEL_HH
#define DUNE_ISTL_BLOCKLEVEL_HH

#include <algorithm>
#include <type_traits>

#include <dune/common/indices.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/hybridutilities.hh>

/*!
 * \file
 * \brief Helper functions for determining the vector/matrix block level
 */

// forward declaration
namespace Dune {
template<typename... Args>
class MultiTypeBlockVector;
template<typename FirstRow, typename... Args>
class MultiTypeBlockMatrix;
} // end namespace Dune

namespace Dune { namespace Impl {

// forward declaration
template<typename T> struct MaxBlockLevel;
template<typename T> struct MinBlockLevel;

//! recursively determine block level of a MultiTypeBlockMatrix
template<typename M, template<typename B> typename BlockLevel, typename Op>
constexpr std::size_t blockLevelMultiTypeBlockMatrix(const Op& op)
{
  // inialize with zeroth diagonal block
  using namespace Dune::Indices;
  using Block00 = typename std::decay_t<decltype(std::declval<M>()[_0][_0])>;
  std::size_t blockLevel = BlockLevel<Block00>::value() + 1;
  // iterate over all blocks to determine min/max block level
  using namespace Dune::Hybrid;
  forEach(integralRange(index_constant<M::N()>()), [&](auto&& i) {
    using namespace Dune::Hybrid; // needed for icc, see issue #31
    forEach(integralRange(index_constant<M::M()>()), [&](auto&& j) {
      using Block = typename std::decay_t<decltype(std::declval<M>()[i][j])>;
      blockLevel = op(blockLevel, BlockLevel<Block>::value() + 1);
    });
  });
  return blockLevel;
}

//! recursively determine block level of a MultiTypeBlockVector
template<typename V, template<typename B> typename BlockLevel, typename Op>
constexpr std::size_t blockLevelMultiTypeBlockVector(const Op& op)
{
  // inialize with zeroth block
  using namespace Dune::Indices;
  using Block0 = typename std::decay_t<decltype(std::declval<V>()[_0])>;
  std::size_t blockLevel = BlockLevel<Block0>::value() + 1;
  // iterate over all blocks to determine min/max block level
  using namespace Dune::Hybrid;
  forEach(integralRange(index_constant<V::size()>()), [&](auto&& i) {
    using Block = typename std::decay_t<decltype(std::declval<V>()[i])>;
    blockLevel = op(blockLevel, BlockLevel<Block>::value() + 1);
  });
  return blockLevel;
}

template<typename T>
struct MaxBlockLevel
{
  static constexpr std::size_t value(){
    if constexpr (IsNumber<T>::value)
      return 0;
    else
      return MaxBlockLevel<typename T::block_type>::value() + 1;
  }
};

template<typename T>
struct MinBlockLevel
{
  // the default implementation assumes minBlockLevel == maxBlockLevel
  static constexpr std::size_t value()
  { return MaxBlockLevel<T>::value(); }
};

// max block level for MultiTypeBlockMatrix
template<typename FirstRow, typename... Args>
struct MaxBlockLevel<Dune::MultiTypeBlockMatrix<FirstRow, Args...>>
{
  static constexpr std::size_t value()
  {
    using M = MultiTypeBlockMatrix<FirstRow, Args...>;
    constexpr auto max = [](const auto& a, const auto& b){ return std::max(a,b); };
    return blockLevelMultiTypeBlockMatrix<M, MaxBlockLevel>(max);
  }
};

// min block level for MultiTypeBlockMatrix
template<typename FirstRow, typename... Args>
struct MinBlockLevel<Dune::MultiTypeBlockMatrix<FirstRow, Args...>>
{
  static constexpr std::size_t value()
  {
    using M = MultiTypeBlockMatrix<FirstRow, Args...>;
    constexpr auto min = [](const auto& a, const auto& b){ return std::min(a,b); };
    return blockLevelMultiTypeBlockMatrix<M, MinBlockLevel>(min);
  }
};

// max block level for MultiTypeBlockVector
template<typename... Args>
struct MaxBlockLevel<Dune::MultiTypeBlockVector<Args...>>
{
  static constexpr std::size_t value()
  {
    using V = MultiTypeBlockVector<Args...>;
    constexpr auto max = [](const auto& a, const auto& b){ return std::max(a,b); };
    return blockLevelMultiTypeBlockVector<V, MaxBlockLevel>(max);
  }
};

// min block level for MultiTypeBlockVector
template<typename... Args>
struct MinBlockLevel<Dune::MultiTypeBlockVector<Args...>>
{
  static constexpr std::size_t value()
  {
    using V = MultiTypeBlockVector<Args...>;
    constexpr auto min = [](const auto& a, const auto& b){ return std::min(a,b); };
    return blockLevelMultiTypeBlockVector<V, MinBlockLevel>(min);
  }
};

// special case: empty MultiTypeBlockVector
template<>
struct MaxBlockLevel<Dune::MultiTypeBlockVector<>>
{
  static constexpr std::size_t value()
  { return 0; };
};

// special case: empty MultiTypeBlockVector
template<>
struct MinBlockLevel<Dune::MultiTypeBlockVector<>>
{
  static constexpr std::size_t value()
  { return 0; };
};

}} // end namespace Dune::Impl

namespace Dune {

//! Determine the maximum block level of a possibly nested vector/matrix type
template<typename T>
constexpr std::size_t maxBlockLevel()
{ return Impl::MaxBlockLevel<T>::value(); }

//! Determine the minimum block level of a possibly nested vector/matrix type
template<typename T>
constexpr std::size_t minBlockLevel()
{ return Impl::MinBlockLevel<T>::value(); }

//! Determine if a vector/matrix has a uniquely determinable block level
template<typename T>
constexpr bool hasUniqueBlockLevel()
{ return maxBlockLevel<T>() == minBlockLevel<T>(); }

//! Determine the block level of a possibly nested vector/matrix type
template<typename T>
constexpr std::size_t blockLevel()
{
  static_assert(hasUniqueBlockLevel<T>(), "Block level cannot be uniquely determined!");
  return Impl::MaxBlockLevel<T>::value();
}

} // end namespace Dune

#endif
