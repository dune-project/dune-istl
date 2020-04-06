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
 * \brief Helper functions for determining the matrix block level
 */

// foward declaration
namespace Dune {
template<typename FirstRow, typename... Args>
class MultiTypeBlockMatrix;
} // end namespace Dune

namespace Dune::Impl {

// forward declaration
template<typename T> struct MaxBlockLevel;
template<typename T> struct MinBlockLevel;

//! recursively determine block level of a MultiTypeBlockMatrix
template<typename M, template<typename B> typename BlockLevel, typename Op>
constexpr int blockLevelMultiTypeBlockMatrix(const Op& op)
{
  // inialize with zeroth diagonal block
  using namespace Dune::Indices;
  using Block00 = typename std::decay_t<decltype(std::declval<M>()[_0][_0])>;
  int blockLevel = BlockLevel<Block00>::value() + 1;
  // iterate over the rest of the blocks to determine min/max block level
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

template<typename T>
struct MaxBlockLevel{
  static constexpr int value(){
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
  static constexpr int value()
  { return MaxBlockLevel<T>::value(); }
};

template<typename FirstRow, typename... Args>
struct MaxBlockLevel<MultiTypeBlockMatrix<FirstRow, Args...>>
{
  static constexpr int value()
  {
    using M = MultiTypeBlockMatrix<FirstRow, Args...>;
    constexpr auto max = [](const auto& a, const auto& b){ return std::max(a,b); };
    return blockLevelMultiTypeBlockMatrix<M, MaxBlockLevel>(max);
  }
};

template<typename FirstRow, typename... Args>
struct MinBlockLevel<MultiTypeBlockMatrix<FirstRow, Args...>>
{
  static constexpr int value()
  {
    using M = MultiTypeBlockMatrix<FirstRow, Args...>;
    constexpr auto min = [](const auto& a, const auto& b){ return std::min(a,b); };
    return blockLevelMultiTypeBlockMatrix<M, MinBlockLevel>(min);
  }
};

} // end namespace Dune::Impl

namespace Dune {

//! Determine the maximum block level of a possibly nested matrix type
template<typename T>
constexpr int maxBlockLevel()
{ return Impl::MaxBlockLevel<T>::value(); }

//! Determine the minimum block level of a possibly nested matrix type
template<typename T>
constexpr int minBlockLevel()
{ return Impl::MinBlockLevel<T>::value(); }

//! Determine if a matrix has a uniquely determinable block level
template<typename T>
constexpr bool hasUniqueBlockLevel()
{ return maxBlockLevel<T>() == minBlockLevel<T>(); }

//! Determine the block level of a possibly nested matrix type
template<typename T>
constexpr int blockLevel()
{
  static_assert(hasUniqueBlockLevel<T>(), "Block level cannot be uniquely determined!");
  return Impl::MaxBlockLevel<T>::value();
}

} // end namespace Dune

#endif
