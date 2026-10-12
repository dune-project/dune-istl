// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_EXPERIMENTAL_INDEXRANGE_HH
#define DUNE_ISTL_EXPERIMENTAL_INDEXRANGE_HH

#include <dune/common/rangeutilities.hh>

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>


/** \file
   \brief Traits for sets of indices.
 */

namespace Dune::Experimental {

namespace Concepts {

/**
 * \brief Requirements for an index container used by IndexedArrayView.
 *
 * The index set must provide random-access traversal over sorted, comparable
 * index values and lookup operations compatible with binary search semantics.
 */
template<class R>
concept OrderedIndexRange =
  std::ranges::random_access_range<R> &&
  std::totally_ordered<std::ranges::range_value_t<R>>;

/**
 * \brief OrderedIndexRange that is safe to use through iterators after moving from
 * temporaries.
 *
 * This encodes a view-like lifetime contract via std::ranges::borrowed_range.
 */
template<class R>
concept BorrowedOrderedIndexRange = OrderedIndexRange<R> && std::ranges::borrowed_range<R>;

} // namespace Concepts

/**
 * \brief Traits for sets of indices.
 *
 * Provides uniform access to the properties of index sets, such as the range of
 * indices covered and the offset of a given index. Specializations for specific
 * index set types can be provided to optimize these operations.
 */
template<Concepts::OrderedIndexRange R>
struct OrderedIndexRangeTraits
{
  //! The type of the index set
  using index_range_type = R;
  //! The type of indices in the index set.
  using index_type = std::ranges::range_value_t<index_range_type>;
  //! The type used to count elements in the index set.
  using size_type = std::ranges::range_size_t<index_range_type>;

  /** \brief Logical index range covered by \p index_set.
   * \details Return the smallest half-open integral range containing
   * all indices in \p index_set.
   */
  constexpr static IntegralRange<index_type> range(const R& index_set)
  {
    auto it = std::cbegin(index_set);
    auto size = std::distance(it, std::cend(index_set));
    if (size == 0)
      return {0, 0};
    return { *it, *(it + (size - 1)) + 1 };
  }

  /** \brief Offset of \p value in the index set.
   * \details Distance to the to the first element in the \p index_set which
   * does not compare less than \p value.
   */
  constexpr static size_type offset(const R& index_set,
                                    const index_type& value)
  {
    const auto lb = std::lower_bound(std::cbegin(index_set), std::cend(index_set), value);
    return static_cast<size_type>(std::distance(std::cbegin(index_set), lb));
  }
};

template<class T>
struct OrderedIndexRangeTraits<IntegralRange<T>>
{
  //! The type of indices in the index set.
  using index_type = T;
  //! The type used to count elements in the index set.
  using size_type = typename IntegralRange<T>::size_type;

  /** \brief Return the logical index range covered by \p index_set.
   *
   * For an IntegralRange, the range is exactly the range of indices it covers.
   */
  constexpr static IntegralRange<index_type> range(
    const IntegralRange<T>& index_set)
  {
    return index_set;
  }

  /** \brief Compute offset of \p value by subtracting the beginning of the range. */
  constexpr static size_type offset(const IntegralRange<T>& index_set,
                                    const index_type& value)
  {
    return static_cast<size_type>(value - *index_set.begin());
  }
};

} // namespace Dune::Experimental

#endif // DUNE_ISTL_EXPERIMENTAL_INDEXRANGE_HH
