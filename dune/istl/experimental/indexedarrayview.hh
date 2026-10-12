// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_EXPERIMENTAL_INDEXEDARRAYVIEW_HH
#define DUNE_ISTL_EXPERIMENTAL_INDEXEDARRAYVIEW_HH

#include "indexrange.hh"
#include "../istlexception.hh"

#include <dune/common/iteratorfacades.hh>
#include <dune/common/rangeutilities.hh>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

/** \file
   \brief Indexed view over random-access data using an explicit index set.
 */

namespace Dune::Experimental {

/**
 * \brief Non-owning indexed view over random-access data.
 *
 * Maps indices from an index set \p R to contiguous values in the data \p T.
 * The view stores both data iterator and index set by value.
 * The data and the index sets are not owned by the view, and must outlive it.
 */
template<std::random_access_iterator T, Concepts::BorrowedOrderedIndexRange R>
class IndexedArrayView
{
public:
  //! The type used for the index set
  using index_range_type = R;
  //! The type of components in the array
  using member_type = std::iter_value_t<T>;
  //! The type used for the index access
  using index_type = typename OrderedIndexRangeTraits<index_range_type>::index_type;
  //! The type used to count elements in the index set
  using size_type = typename OrderedIndexRangeTraits<index_range_type>::size_type;
  //! The type used for references to the components
  using reference = std::iter_reference_t<T>;
  //! The type used for const references to the components
#if __cpp_lib_ranges_as_const >= 202207L
  using const_reference = std::iter_const_reference_t<T>;
#else
  using const_reference = std::common_reference_t<const std::iter_value_t<T>&&,
                                                  std::iter_reference_t<T>>;
#endif

  /**
   * \brief Access element by an index from the index set.
   * \param i Index from the index set.
   * \throws ISTLError if bounds checking is enabled and \p i is not present.
   */
  reference operator[](index_type i)
  {
    const size_type offset = OrderedIndexRangeTraits<index_range_type>::offset(index_set_, i);
#ifdef DUNE_ISTL_WITH_CHECKING
    if (offset >= size() || index_set_[offset] != i)
      DUNE_THROW(ISTLError, "index " << i << " not in base array index set");
#endif
    return data_[offset];
  }

  /**
   * \brief Const access element by an index from the index set.
   * \param i Index from the index set.
   * \throws ISTLError if bounds checking is enabled and \p i is not present.
   */
  const_reference operator[](index_type i) const
  {
    const size_type offset = OrderedIndexRangeTraits<index_range_type>::offset(index_set_, i);
#ifdef DUNE_ISTL_WITH_CHECKING
    if (offset >= size() || index_set_[offset] != i)
      DUNE_THROW(ISTLError, "index " << i << " not in base array index set");
#endif
    return data_[offset];
  }

private:
  class ConstIterator;

  /** \brief Mutable random-access iterator over IndexedArrayView. */
  class Iterator
    : public Dune::IteratorFacade<Iterator,
                                  std::random_access_iterator_tag,
                                  std::remove_reference_t<reference>,
                                  reference>
  {
    using Facade = Dune::IteratorFacade<Iterator,
                                        std::random_access_iterator_tag,
                                        std::remove_reference_t<reference>,
                                        reference>;

  public:
    //! The type used to compute offsets in the index set
    using offset_type = typename OrderedIndexRangeTraits<index_range_type>::size_type;

    //! Default constructor.
    constexpr Iterator() noexcept = default;
    //! Copy constructor.
    constexpr Iterator(const Iterator&) = default;
    //! Copy assignment operator.
    constexpr Iterator& operator=(const Iterator&) = default;

    /** \brief Construct an iterator from the data, index set, and offset. */
    explicit constexpr Iterator(
      T data,
      index_range_type index_set_,
      offset_type
        offset) noexcept(std::is_nothrow_copy_constructible_v<T> &&
                         std::is_nothrow_copy_constructible_v<index_range_type> &&
                         std::is_nothrow_copy_constructible_v<offset_type>)
      : data_(data)
      , index_set_(index_set_)
      , offset_(offset)
    {
    }

    //! The index in the index set corresponding to the current iterator position.
    constexpr index_type index() const { return index_set_[offset_]; }
    //! The offset in the index set corresponding to the current iterator position.
    constexpr offset_type offset() const { return offset_; }

    // Reference to the component at the current iterator position.
    constexpr typename Facade::reference operator*() const { return data_[offset_]; }

  private:
    friend Dune::IteratorFacadeAccess;
    friend ConstIterator;

    constexpr const offset_type& baseIterator() const { return offset_; }
    constexpr offset_type& baseIterator() { return offset_; }

    T data_;
    index_range_type index_set_;
    offset_type offset_;
  };

  /** \brief Const random-access iterator over IndexedArrayView. */
  class ConstIterator
    : public Dune::IteratorFacade<ConstIterator,
                                  std::random_access_iterator_tag,
                                  std::remove_reference_t<const_reference>,
                                  const_reference>
  {
    using Facade =
      Dune::IteratorFacade<ConstIterator,
                           std::random_access_iterator_tag,
                           std::remove_reference_t<const_reference>,
                           const_reference>;

  public:
    //! The type used to compute offsets in the index set
    using offset_type = typename OrderedIndexRangeTraits<index_range_type>::size_type;

    //! Default constructor.
    constexpr ConstIterator() noexcept = default;
    //! Copy constructor.
    constexpr ConstIterator(const ConstIterator& other) = default;
    //! Copy assignment operator.
    constexpr ConstIterator& operator=(const ConstIterator& other) = default;

    //! Construct a const iterator from a mutable iterator.
    constexpr ConstIterator(const Iterator& it) noexcept(std::is_nothrow_copy_constructible_v<Iterator>)
      : it_(it)
    {}

    //! Copy assignment operator from a mutable iterator.
    constexpr ConstIterator& operator=(const Iterator& it) noexcept(std::is_nothrow_copy_assignable_v<Iterator>)
    {
      it_ = it;
      return *this;
    }

    //! The index in the index set corresponding to the current iterator position.
    constexpr index_type index() const { return it_.index(); }
    //! The offset in the index set corresponding to the current iterator position.
    constexpr offset_type offset() const { return it_.offset(); }

    //! Reference to the component at the current iterator position.
    constexpr typename Facade::reference operator*() const { return *it_; }

  private:
    friend Dune::IteratorFacadeAccess;
    friend Iterator;

    constexpr const offset_type& baseIterator() const { return it_.baseIterator(); }
    constexpr offset_type& baseIterator() { return it_.baseIterator(); }

    Iterator it_;
  };

public:
  //! Iterator type with mutable access to the underlying data.
  using iterator = Iterator;
  //! Iterator type with const access to the underlying data.
  using const_iterator = ConstIterator;

  //! Iterator to the first element in the view.
  constexpr iterator begin() noexcept(
    noexcept(iterator(data_, index_set_, size_type{})))
  {
    return iterator(data_, index_set_, size_type{ 0 });
  }

  //! Iterator to one past the last element in the view.
  constexpr iterator end() noexcept(
    noexcept(iterator(data_, index_set_, size_type{})))
  {
    return iterator(data_, index_set_, size());
  }

  /** \brief Find element with index \p i in the index set.
   * \returns Iterator to the element with index \p i, or end() if \p i is not present.
  */
  iterator find(size_type i)
  {
    const size_type ofs = OrderedIndexRangeTraits<index_range_type>::offset(index_set_, i);
    return (ofs == size() || std::cmp_not_equal(index_set_[ofs], i)) ? end() : iterator(data_, index_set_, ofs);
  }

  bool contains(index_type i) const
  {
    const size_type ofs = OrderedIndexRangeTraits<index_range_type>::offset(index_set_, i);
    return ofs != size() && index_set_[ofs] == i;
  }

  //! Const iterator to the first element in the view.
  constexpr const_iterator begin() const
    noexcept(noexcept(iterator(data_, index_set_, size_type{})))
  {
    return iterator(data_, index_set_, size_type{ 0 });
  }

  //! Const iterator to one past the last element in the view.
  constexpr const_iterator end() const
    noexcept(noexcept(iterator(data_, index_set_, size_type{})))
  {
    return iterator(data_, index_set_, size());
  }

  /** \brief Find element with index \p i in the index set.
   * \returns Const iterator to the element with index \p i, or end() if \p i is not present.
  */
  const_iterator find(size_type i) const
  {
    const size_type ofs = OrderedIndexRangeTraits<index_range_type>::offset(index_set_, i);
    return (ofs == size() || index_set_[ofs] != i) ? end() : const_iterator{iterator{data_, index_set_, ofs}};
  }

  //===== sizes

  //! The number of elements in the view, i.e. the size of the index set.
  constexpr size_type size() const
    noexcept(noexcept(std::ranges::size(index_set_)))
  {
    return std::ranges::size(index_set_);
  }

  //! The logical index range covered by the view, as determined by the index set.
  constexpr index_range_type indexSet() const
  {
    return index_set_;
  }

  /**
   * \brief Construct a view from data iterator and index set.
   *
   * \param data Iterator to first element of the underlying contiguous data.
   * \param index_set Set or view of valid external indices.
   */
  IndexedArrayView(T data, index_range_type index_set) noexcept(
    std::is_nothrow_copy_constructible_v<T> &&
    std::is_nothrow_copy_constructible_v<index_range_type>)
    : data_(data)
    , index_set_(index_set) {};

private:
  T data_;
  index_range_type index_set_;
};

} // namespace Dune::Experimental

#endif // DUNE_ISTL_EXPERIMENTAL_INDEXEDARRAYVIEW_HH
