// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_EXPERIMENTAL_SPARSEINDEXRANGES_HH
#define DUNE_ISTL_EXPERIMENTAL_SPARSEINDEXRANGES_HH

#include "indexrange.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/classname.hh>
#include <dune/common/rangeutilities.hh>

#include <dune/common/std/type_traits.hh>
#include <dune/common/std/no_unique_address.hh>

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <utility>
#include <variant>
#include <iostream>
#include <functional>

namespace Dune::Experimental
{

namespace Impl
{
  //! Tag base used to opt IndexRangeView into std::ranges::borrowed_range via the constrained partial specialization below.
  struct SparseIndexRangesBase {};

  /**
   * @brief Threshold (in bytes) for switching from linear to binary search.
   *
   * Sparse index ranges smaller than this threshold use linear search.
   * Sparse index ranges larger or equal to this threshold use binary search.
   *
   * Override this by defining DUNE_ISTL_SPARSE_SEARCH_THRESHOLD before including
   * this header, or by passing -DDUNE_ISTL_SPARSE_SEARCH_THRESHOLD=<bytes> to the
   * compiler. Note that different compilers and architectures may have different
   * optimal thresholds, so tuning may be beneficial for performance.
   * The default value is set to 0 bytes, which effectively disables linear search
   * and uses binary search for all sparse index range sizes.
   */
#if !defined(DUNE_ISTL_SPARSE_SEARCH_THRESHOLD)
  static constexpr std::size_t SPARSE_SEARCH_THRESHOLD = 0;
#else
  static constexpr std::size_t SPARSE_SEARCH_THRESHOLD = DUNE_ISTL_SPARSE_SEARCH_THRESHOLD;
#endif

  template<class A>
  struct SparseIndexRangeHelper {

    //! Type of the allocator used for memory management.
    using allocator_type = A;
    //! Traits of the allocator type.
    using alloc_traits = std::allocator_traits<allocator_type>;

    //! Type of the size parameters
    using size_type = typename std::allocator_traits<allocator_type>::size_type;
    //! Type of the offset storage allocator.
    using size_alloc_type = typename alloc_traits::template rebind_alloc<typename alloc_traits::size_type>;
    //! Traits of the offset storage allocator.
    using size_alloc_traits = std::allocator_traits<size_alloc_type>;

    //! Type of the indices
    using index_type = uint_fast64_t;

    //! Variant type to hold pointers to index storage of different integer types.
    using IndexVariant = std::variant<uint_least64_t *, uint_least32_t *, uint_least16_t *, uint_least8_t *>;

    //! Custom scope deleter for index storage pointers, which deallocates upon destruction.
    template<class OtherAlloc>
    struct DeallocationGuard;

    //! Debug function to print the current state of offsets and indices for testing and development purposes.
    inline void debug(std::span<size_type> offsets, IndexVariant indices);

    //! Allocate and initialize storage for indices variant. Returns index variant and allocated size
    static constexpr std::pair<IndexVariant,size_type> initAllocateIndices(const allocator_type &allocator, size_type count, size_type max_size);
    //! Allocate and initialize storage for offsets. Returns pointer to offsets
    static constexpr size_type* initAllocateOffsets(const allocator_type &allocator, size_type size);
    //! Deallocates storage for indices and offsets
    static constexpr void deallocate(const allocator_type& allocator, IndexVariant indices, size_type* offsets, size_type size);

    //! Compresses the index storage by removing invalid indices and switching to a smaller integer type if possible. Returns new index variant with compressed storage and updates offsets. Old indices are deallocated.
    template<class V = IndexVariant>
    static constexpr V compress(std::span<size_type> offsets, IndexVariant indices, const allocator_type& allocator);

    //! Adds index i to the j-th index set
    template<class index_t>
    static constexpr bool addIndex(std::span<index_t> index_set, index_t j, std::invocable auto grow_index_set_fn);

    //! Grows the storage for the indices of the index sets. Returns new index variant with grown storage and updates offsets. Old indices are deallocated.
    static constexpr void growIndices(std::span<size_type> offsets, IndexVariant& indices, size_type new_count, const allocator_type& allocator, auto uninitialized_copy_fn);

    //! Grows the storage for the indices of the last index set. Returns new index variant with grown storage and updates offsets. Old indices are deallocated.
    static constexpr void growLastIndexRange(std::span<size_type> offsets, IndexVariant& indices, double growth_factor, const allocator_type& allocator);

    //! Grows storage for all index sets using a global fairness policy.
    //!
    //! Policy:
    //! - Let m be the maximum number of used entries among all index sets.
    //! - Let f be the requested growth factor (> 1).
    //! - Every index set receives approximately the same additive slack,
    //!   computed from the densest set as max(1, floor(m * (f - 1))).
    //!
    //! Rationale:
    //! - Random insertion across index sets: keeps free capacity roughly homogeneous,
    //!   preserving amortized insertion complexity.
    //! - Sequenced insertion (index sets filled in order): avoids a strict per index set policy
    //!   that would trigger frequent reallocations as each new index set becomes active.
    //!
    //! Trade-off:
    //! - If final index sets densities are highly skewed, this can over-allocate compared
    //!   to per index set growth, but it reduces reallocation churn in common uniform cases.
    //!
    //! Returns new index variant with grown storage and updates offsets.
    //! Old indices are deallocated.
    static constexpr void growAllIndexRanges(std::span<size_type> offsets, IndexVariant& indices, double growth_factor, const allocator_type& allocator);

    //! Finds the position of index j in the index set. Uses linear search for small index sets
    //! and binary search for large index sets. The threshold is architecture-tuned and can be
    //! overridden via DUNE_ISTL_SPARSE_SEARCH_THRESHOLD compiler flag.
    template<class IS>
    static constexpr auto lowerBound(IS index_set, auto j) {
      using value_type = std::ranges::range_value_t<IS>;
      // Use portable architecture-aware threshold from config header
      if (index_set.size() * sizeof(value_type) <= Impl::SPARSE_SEARCH_THRESHOLD)
        return std::find_if(std::cbegin(index_set), std::cend(index_set), [j](value_type index){ return index >= j; });
      else
        return std::lower_bound(std::cbegin(index_set), std::cend(index_set), j);
    }

    //! Finds the position of the first index greater than j in the index set. Uses linear search
    //! for small index sets and binary search for large index sets. See lowerBound() for threshold details.
    template<class IS>
    static constexpr auto upperBound(IS index_set, auto j) {
      using value_type = std::ranges::range_value_t<IS>;
      // Use portable architecture-aware threshold from config header
      if (index_set.size() * sizeof(value_type) <= Impl::SPARSE_SEARCH_THRESHOLD)
        return std::find_if(std::cbegin(index_set), std::cend(index_set), [j](value_type index){ return index > j; });
      else
        return std::upper_bound(std::cbegin(index_set), std::cend(index_set), j);
    }

  private:
    template<class V, std::size_t... I>
    static constexpr V makeIndexVariantWithAlternative(std::size_t selected, std::index_sequence<I...>)
    {
      V result{};
      const bool assigned = ((selected == I ? (result = V{std::in_place_index<I>, nullptr}, true) : false) || ...);
      if (!assigned)
        DUNE_THROW(RangeError, "Selected index variant alternative is out of range: " << selected);
      return result;
    }

    template<class V, std::size_t... I>
    static constexpr std::size_t selectSmallestIndexStorage(size_type max_index, std::index_sequence<I...>)
    {
      constexpr std::size_t variant_size = sizeof...(I);
      constexpr std::size_t invalid = variant_size;
      std::size_t best = invalid;
      std::size_t best_size = std::numeric_limits<std::size_t>::max();
      std::size_t fallback = 0;
      size_type fallback_max = 0;

      auto consider = [&]<std::size_t J>() {
        using pointer_type = std::variant_alternative_t<J, V>;
        static_assert(std::is_pointer_v<pointer_type>, "Variant alternatives must be pointer types");
        using index_t = std::remove_pointer_t<pointer_type>;
        static_assert(std::is_integral_v<index_t>, "Variant pointer element types must be integral");

        const auto max_supported = static_cast<size_type>(std::numeric_limits<index_t>::max());
        if (max_supported >= fallback_max) {
          fallback = J;
          fallback_max = max_supported;
        }

        if (max_index <= max_supported) {
          constexpr std::size_t storage_size = sizeof(index_t);
          if (storage_size < best_size) {
            best = J;
            best_size = storage_size;
          }
        }
      };

      (consider.template operator()<I>(), ...);

      if (best != invalid)
        return best;

      assert(false && "Variant V does not provide an index type that can represent max_index");
      return fallback;
    }

    template<class V>
    static constexpr V makeSmallestIndexVariant(size_type max_index)
    {
      constexpr std::size_t variant_size = std::variant_size_v<V>;
      static_assert(variant_size > 0, "Variant V must not be empty");
      const auto selected = selectSmallestIndexStorage<V>(max_index, std::make_index_sequence<variant_size>{});
      return makeIndexVariantWithAlternative<V>(selected, std::make_index_sequence<variant_size>{});
    }
  };

} // namespace Impl


  template<class I, class A>
  class SparseIndexRanges;

  template<class A>
  class CompressedSparseIndexRanges;

  // ---------------------------------------------------------------------------
  // UnsequencedSparseIndexRangeBuilder declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Builder that allows inserting indices into any index set in any order.
   *
   * @par Build sequence
   * 1. Optional. Call `setSize(i, s)` or `incrementSize(i, delta)` for every index set i.
   * 2. Call `addIndex(i, j)` for each `(set i, index j)` pair in any order. Duplicates are ignored.
   * 4. Construct a `SparseIndexRanges` or `CompressedSparseIndexRanges` using their constructors.
   *
   * The builder is neither copyable nor movable. Its storage is transferred to the
   * index ranges once built.
   */
  template<class A = std::allocator<void>>
  class UnsequencedSparseIndexRangeBuilder
  {
    enum Stage : uint_least8_t
    {
      BuildSizes,
      BuildSetIndices,
      Invalid
    };

    using IndexVariant = typename Impl::SparseIndexRangeHelper<A>::IndexVariant;

  public:

    using allocator_type = A;

    //! Type of the size parameters
    using size_type = typename std::allocator_traits<allocator_type>::size_type;

    //! Type of the indices
    using index_type = uint_fast64_t;

    /**
     * @brief Construct an unsequenced builder for @p size index sets with index range [0, @p is_size).
     * @param avg_indices Expected average number of indices per set, used to initialize reserved sizes for index sets. This is a performance hint.
     */
    constexpr UnsequencedSparseIndexRangeBuilder(size_type size, size_type is_size, size_type avg_indices = 0, allocator_type allocator = allocator_type());
    constexpr ~UnsequencedSparseIndexRangeBuilder();

    constexpr UnsequencedSparseIndexRangeBuilder(const UnsequencedSparseIndexRangeBuilder &) = delete;
    constexpr UnsequencedSparseIndexRangeBuilder(UnsequencedSparseIndexRangeBuilder &&) = delete;
    constexpr UnsequencedSparseIndexRangeBuilder &operator=(const UnsequencedSparseIndexRangeBuilder &) = delete;
    constexpr UnsequencedSparseIndexRangeBuilder &operator=(UnsequencedSparseIndexRangeBuilder &&) = delete;

    //! Set the reserved size of the i-th index set to @p s. Must be called before `addIndex()`.
    constexpr void setSize(size_type i, size_type s);
    //! Increment the reserved size of the i-th index set by @p increment. Must be called before `addIndex()`.
    constexpr void incrementSize(size_type i, size_type increment = 1);
    //! Insert index @p j into index set @p i. Duplicate insertions are silently ignored.
    constexpr void addIndex(size_type i, index_type j);

  private:
    //! Finalize all sizes and allocate index storage.
    constexpr void endSizes();

    template<class OtherAlloc>
    friend class CompressedSparseIndexRanges;
    template<class I, class OtherAlloc>
    friend class SparseIndexRanges;

    size_type size_;
    size_type is_size_;
    size_type *offsets_;
    IndexVariant indices_;
    DUNE_NO_UNIQUE_ADDRESS allocator_type alloc_;
    Stage stage_;
  };

  // ---------------------------------------------------------------------------
  // SequencedSparseIndexRangeBuilder declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Builder that fills index sets one at a time, in sequential order.
   *
   * @details Use this builder when index sets are filled in order from first to last
   * (e.g., processing rows of a sparse matrix from top to bottom).
   *
   * @par Build sequence
   * Obtain an `IndexRangeBuilder` from `begin()` and advance it with `operator++()` after
   * finishing each index set. A range-based for loop is the most convenient pattern:
   * @code
   *   SequencedSparseIndexRangeBuilder sb(rows, cols);
   *   for (auto& isb : sb) {
   *     isb.insert(j1);
   *     isb.insert(j2);
   *   }
   *   SparseIndexRanges sis(std::move(sb));
   * @endcode
   * Duplicate insertions within a set are silently ignored. After all sets are filled,
   * construct a `SparseIndexRanges` or `CompressedSparseIndexRanges` using their constructors.
   *
   * The builder is neither copyable nor movable. Its storage is transferred to the
   * index ranges once built.
   */
  template<class A = std::allocator<void>>
  class SequencedSparseIndexRangeBuilder
  {
    enum Stage : uint_least8_t
    {
      Allocated,
      BuildingIndexRanges,
      Built,
      Invalid
    };

    using IndexVariant = typename Impl::SparseIndexRangeHelper<A>::IndexVariant;

  public:

    using allocator_type = A;

    //! Type of the size parameters
    using size_type = typename std::allocator_traits<allocator_type>::size_type;

    //! Type of the indices
    using index_type = uint_fast64_t;

    //! Construct a builder for @p size index sets with index range [0, @p is_size).
    //! @param avg_indices Expected average number of indices per set; used to pre-allocate storage.
    constexpr SequencedSparseIndexRangeBuilder(size_type size, size_type is_size, size_type avg_indices = 2048, allocator_type allocator = allocator_type());

    constexpr ~SequencedSparseIndexRangeBuilder();

    constexpr SequencedSparseIndexRangeBuilder(const SequencedSparseIndexRangeBuilder &) = delete;
    constexpr SequencedSparseIndexRangeBuilder(SequencedSparseIndexRangeBuilder &&) = delete;
    constexpr SequencedSparseIndexRangeBuilder &operator=(const SequencedSparseIndexRangeBuilder &) = delete;
    constexpr SequencedSparseIndexRangeBuilder &operator=(SequencedSparseIndexRangeBuilder &&) = delete;

    struct EndSentinel {};

    /**
     * @brief Handle for inserting indices into the current index set of a `SequencedSparseIndexRangeBuilder`.
     *
     * @details Obtained from `SequencedSparseIndexRangeBuilder::begin()`. Use `operator++()` to commit the
     * current index set and advance to the next one. The iterator pattern enables driving
     * the builder via a range-based for loop.
     *
     * Advancing past the last index set marks the builder as finished, enabling detection via comparison with `EndSentinel{}`.
     * After finishing, construct a `SparseIndexRanges` or `CompressedSparseIndexRanges` using their constructors.
     * The `IndexRangeBuilder` must not be used after the enclosing `SequencedSparseIndexRangeBuilder` is destroyed.
     */
    class IndexRangeBuilder
    {
      constexpr IndexRangeBuilder(SequencedSparseIndexRangeBuilder* builder);

    public:
      constexpr IndexRangeBuilder(const IndexRangeBuilder &) = delete;
      constexpr IndexRangeBuilder &operator=(const IndexRangeBuilder &) = delete;

      constexpr IndexRangeBuilder(IndexRangeBuilder && other);
      constexpr IndexRangeBuilder &operator=(IndexRangeBuilder && other);

      constexpr ~IndexRangeBuilder() = default;

      //! Insert index @p j into the current index set. Duplicate insertions are silently ignored.
      constexpr void insert(index_type j);

      //! Returns true if index @p j has already been inserted into the current index set.
      constexpr bool contains(index_type j) const;

      /**
       * @brief Replace the current index set with the indices in [begin, end).
       *
       * @details This overwrites any indices previously inserted into the current
       * index set. The provided range is copied into the active storage of the
       * current row and the remainder of that row buffer is reset to the invalid
       * sentinel value, so old entries do not remain observable through
       * contains(), lower_bound(), or later compression.
       *
       * If the current row buffer is too small, it is grown before the new
       * indices are copied. When @p do_sort is true, the copied prefix is sorted
       * in ascending order. When @p do_unique is true, duplicate values are
       * removed from that sorted prefix. The final logical size of the current
       * row is the size after optional sorting and deduplication.
       *
       * @param begin Iterator to the first replacement index.
       * @param end Sentinel one past the last replacement index.
       * @param do_sort Whether to sort the copied indices before storing them.
       * @param do_unique Whether to remove duplicate indices after the optional
       *                  sort. This assumes equal elements become adjacent.
       */
      template<std::forward_iterator It, std::sentinel_for<It> End>
      constexpr void overrideIndices(It begin, End end, bool do_sort = true, bool do_unique = true);

      //! Returns the zero-based number of the index set currently being filled.
      constexpr size_type index() const;

      //! Commit the current index set and advance to the next one.
      //! After advancing past the last set, comparing with `EndSentinel{}` returns true.
      constexpr IndexRangeBuilder& operator++();

      //! Returns the number of distinct indices inserted into the current index set so far.
      constexpr size_type size() const;

      //! Returns true when all index sets have been filled (used to detect the end sentinel).
      constexpr friend bool operator==(const IndexRangeBuilder& is_builder, EndSentinel) {
        assert(is_builder.stage_ != Stage::Invalid);
        return is_builder.stage_ == Stage::Built;
      }

      //! Dereference — returns a reference to this `IndexRangeBuilder` itself (enables range-for).
      constexpr IndexRangeBuilder& operator*();

    private:
      friend class SequencedSparseIndexRangeBuilder;
      SequencedSparseIndexRangeBuilder* builder_;
      size_type i_;
      size_type is_count_; // number of indices inserted in the current index set
      Stage stage_; // stage of this builder, used to check end sentinel without relying on the address of the builder (which may be moved after the last increment).
    };

    //! Returns an `IndexRangeBuilder` positioned at the first index set.
    constexpr IndexRangeBuilder begin();

    //! Returns the end sentinel used to detect when all index sets have been filled.
    constexpr EndSentinel end();
  private:

    template<class OtherAlloc>
    friend class CompressedSparseIndexRanges;
    template<class I, class OtherAlloc>
    friend class SparseIndexRanges;
    size_type size_;
    size_type is_size_;
    size_type *offsets_;
    IndexVariant indices_;
    DUNE_NO_UNIQUE_ADDRESS allocator_type alloc_;
    Stage stage_;
  };


  /**
   * @brief Compact storage of multiple sorted sparse index sets sharing a single allocation.
   *
   * @details A `SparseIndexRanges<A>` stores `size()` sorted sparse index sets. Each index set
   * is a subset of the integer range given by `range()`.
   *
   * @par Construction
   * Objects are built via one of two builder types and cannot be default-constructed:
   * - `SequencedSparseIndexRangeBuilder`: fill index sets sequentially, one at a time from first to last.
   * - `UnsequencedSparseIndexRangeBuilder`: insert indices across all sets in any order, optionally, declaring sizes first.
   *
   * After construction the storage is compressed: unused capacity is discarded and indices are stored contiguously.
   *
   * @par Access
   * Iterate over all index sets with `begin()`/`end()`, or access individual sets with
   * `operator[]`. Each element is an `IndexRangeView`, a lightweight read-only range over
   * the sorted indices of a single index set.
   *
   * @par Ownership
   * The object is move-only; copy construction and copy assignment are disabled.
   *
   * @tparam A Allocator type (default: `std::allocator<I>`).
   */
  template <class I = std::uint_least32_t, class A = std::allocator<I>>
  class SparseIndexRanges
  {
    using alloc_type = A;
    using alloc_traits = std::allocator_traits<alloc_type>;

    using size_alloc_type = typename alloc_traits::template rebind_alloc<typename alloc_traits::size_type>;
    using size_alloc_traits = std::allocator_traits<size_alloc_type>;

    // Nested type declarations
    class IndexIterator;

    constexpr SparseIndexRanges() = delete;
    constexpr SparseIndexRanges(const SparseIndexRanges &other) = delete;
    constexpr SparseIndexRanges &operator=(const SparseIndexRanges &other) = delete;

  public:

    //! Type to access the index sets
    using size_type = typename alloc_traits::size_type;

    //! Type of the indices
    using index_type = I;

    //! Allocator type used for memory management
    using allocator_type = A;

    //! Iterator over SparseIndexRanges, iterates over the view of index sets (e.g., rows of the matrix)
    class Iterator;

    //! View of an index set, provides access to the indices of a single index set (e.g., column indices of a single row)
    class IndexRangeView;

    //! Construct from a finished `SequencedSparseIndexRangeBuilder`, taking ownership of its storage and compressing it.
    template<class OtherAlloc>
    requires std::constructible_from<A, OtherAlloc>
    constexpr SparseIndexRanges(SequencedSparseIndexRangeBuilder<OtherAlloc>&& builder);

    //! Construct from a finished `UnsequencedSparseIndexRangeBuilder`, taking ownership of its storage and compressing it.
    template<class OtherAlloc>
    requires std::constructible_from<A, OtherAlloc>
    constexpr SparseIndexRanges(UnsequencedSparseIndexRangeBuilder<OtherAlloc>&& builder);

    //! Move-construct, transferring ownership of storage from @p other. @p other is left in a valid but empty state.
    constexpr SparseIndexRanges(SparseIndexRanges &&other);

    //! Move-assign, releasing current storage and transferring ownership from @p other. @p other is left in a valid but empty state.
    constexpr SparseIndexRanges &operator=(SparseIndexRanges &&other);

    //! Swap contents with @p other honoring allocator swap propagation traits.
    constexpr void swap(SparseIndexRanges &other);

    //! Swap contents of @p lhs and @p rhs, honoring allocator swap propagation traits.
    friend constexpr void swap(SparseIndexRanges &lhs, SparseIndexRanges &rhs)
    {
      lhs.swap(rhs);
    }

    //! Destroy the index sets and release all allocated memory.
    constexpr ~SparseIndexRanges();

    //! Returns the number of index sets (e.g., the number of rows in a sparse matrix).
    constexpr size_type size() const noexcept;

    //! Returns the total number of indices stored across all index sets (sum of all individual set sizes).
    constexpr size_type count() const;

    //! Returns the integer range [0, n) that every stored index belongs to.
    constexpr IntegralRange<index_type> range() const noexcept;

    //! Returns the flat-storage offset of the start of the i-th index set.
    //! @param i Index-set number; 0 <= i <= size(). Pass i == size() to obtain the total index count.
    constexpr size_type offset(size_type i) const;

    //! Returns a read-only view of the i-th index set.
    //! @param i Index-set number; must satisfy 0 <= i < size().
    constexpr IndexRangeView operator[](size_type i) const { return {this, i}; };

    //! Returns an iterator to the first index set (index set 0).
    constexpr Iterator begin() const { return {this, 0}; }

    //! Returns an iterator past the last index set.
    constexpr Iterator end() const { return {this, size()}; }

    //! Writes all index sets to @p os, one per line in the form "[i]: j0 j1 ...". Returns @p os.
    std::ostream &print(std::ostream &os) const;

    //! Returns true if @p p1 and @p p2 contain identical index sets with identical values.
    constexpr friend bool operator==(const SparseIndexRanges &p1, const SparseIndexRanges &p2)
    {
      if (std::addressof(p1) == std::addressof(p2))
        return true;

      if (p1.size_ != p2.size_ || p1.is_size_ != p2.is_size_)
        return false;

      if (!std::equal(p1.offsets_, p1.offsets_ + p1.size_ + 1, p2.offsets_))
        return false;

      return std::equal(p1.indices_, p1.indices_ + p1.count(), p2.indices_);;
    }

  private:

    size_type size_; //! number of index sets, e.g., rows of the matrix
    index_type is_size_; //! size of the index sets, e.g., number of columns of the matrix
    size_type *offsets_; //! offsets to the start of each index set in the flat index storage
    index_type *indices_; //! flat storage of all indices across all index sets
    DUNE_NO_UNIQUE_ADDRESS alloc_type alloc_;
  };

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::Iterator declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Random-access iterator over the index sets of a `SparseIndexRanges`.
   *
   * @details Dereferencing yields an `IndexRangeView` for the current index set.
   * Obtained via `SparseIndexRanges::begin()` and `SparseIndexRanges::end()`.
   * Supports the full random-access iterator interface via `Dune::IteratorFacade`.
   *
   * The iterator is invalidated if the underlying `SparseIndexRanges` is destroyed or moved.
   */
  template <class I, class A>
  class SparseIndexRanges<I,A>::Iterator
      : public Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, IndexRangeView, IndexRangeView, Dune::ProxyArrowResult<IndexRangeView>>
  {
    using Facade = Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, IndexRangeView, IndexRangeView, Dune::ProxyArrowResult<IndexRangeView>>;

    constexpr Iterator(SparseIndexRanges const *sp, size_type i);

  public:
    using reference = typename Facade::reference;
    using difference_type = typename Facade::difference_type;

    constexpr Iterator() = default;

    constexpr reference operator*() const;

  private:
    friend Dune::IteratorFacadeAccess;
    friend SparseIndexRanges;

    constexpr const size_type &baseIterator() const;

    constexpr size_type &baseIterator();

    SparseIndexRanges const *cis_;
    size_type i_;
  };

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::IndexIterator declaration
  // ---------------------------------------------------------------------------

  template <class I, class A>
  class SparseIndexRanges<I,A>::IndexIterator
      : public Dune::IteratorFacade<IndexIterator, std::random_access_iterator_tag, index_type, index_type, Dune::ProxyArrowResult<index_type>>
  {
    using Facade = Dune::IteratorFacade<IndexIterator, std::random_access_iterator_tag, index_type, index_type, Dune::ProxyArrowResult<index_type>>;

  public:
    using reference = typename Facade::reference;
    using difference_type = typename Facade::difference_type;

    constexpr IndexIterator(SparseIndexRanges const *csi, size_type i_, size_type j);

    constexpr IndexIterator() = default;

    constexpr size_type offset() const;

    constexpr reference operator*() const;

  private:
    friend Dune::IteratorFacadeAccess;

    constexpr const size_type &baseIterator() const;

    constexpr size_type &baseIterator();

    SparseIndexRanges const *cis_;
    size_type i_;
    size_type j_;
  };

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::IndexRangeView declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Read-only view of a single index set inside a `SparseIndexRanges`.
   *
   * @details Provides a `std::ranges::view` over the sorted indices of one index set.
   * Obtained via `SparseIndexRanges::operator[]` or by dereferencing a `SparseIndexRanges::Iterator`.
   *
   * `IndexRangeView` is a lightweight proxy (pointer + index offset) that is cheap to copy.
   * It becomes invalid if the underlying `SparseIndexRanges` is destroyed or moved.
   */
  template <class I, class A>
  class SparseIndexRanges<I,A>::IndexRangeView
      : public Impl::SparseIndexRangesBase
  {
    friend class SparseIndexRanges<I,A>::Iterator;
    friend class SparseIndexRanges<I,A>;

    constexpr IndexRangeView(const SparseIndexRanges *cis, size_type i);

  public:
    IndexRangeView() = default;
    IndexRangeView(const IndexRangeView&) = default;
    IndexRangeView& operator=(const IndexRangeView&) = default;

    //! Type of the indices
    using index_type = typename SparseIndexRanges<I,A>::index_type;

    using size_type = typename SparseIndexRanges<I,A>::size_type;

    IndexRangeView(IndexRangeView&&) = default;
    IndexRangeView& operator=(IndexRangeView&&) = default;

    //! Returns an iterator to the first index in this index set.
    constexpr IndexIterator begin() const;
    //! Returns an iterator past the last index in this index set.
    constexpr IndexIterator end() const;

    //! Returns the number of indices in this index set.
    constexpr size_type size() const;

    //! Returns true if this index set is empty (i.e., contains no indices).
    constexpr bool empty() const { return size() == 0; }

    //! Returns the first index in this index set. Precondition: the set is not empty.
    constexpr index_type front() const { return *begin(); }

    //! Returns the last index in this index set. Precondition: the set is not empty.
    constexpr index_type back() const { return *(end() - 1); }

    //! Returns the index at local position @p i in this index set. Precondition: i < size().
    constexpr index_type operator[](size_type i) const { return *(begin() + i); }

    //! Returns an integer range [0, n) usable as a local index range over this set.
    constexpr IntegralRange<index_type> range() const;

    //! Returns true if index @p j is contained in this index set. Complexity: O(size()) for small sets, O(log(size())) for large sets.
    constexpr bool contains(index_type j) const;

    //! Returns an iterator to the first index >= @p i in this index set, or `end()` if none exists.
    constexpr IndexIterator lower_bound(index_type i) const;


    constexpr IndexIterator upper_bound(index_type i) const;


    constexpr IndexIterator find(index_type i) const;

    //! Calls @p f(j) for every index @p j in this index set, in ascending order.
    template<std::invocable<index_type> F>
    constexpr void for_each(F &&f) const
    {
      const std::span<const index_type> index_set(cis_->indices_ + cis_->offset(i_), cis_->indices_ + cis_->offset(i_ + 1));
      std::for_each(std::cbegin(index_set), std::cend(index_set), std::forward<F>(f));
    }

  private:
    SparseIndexRanges const *cis_;
    size_type i_;
  };

  /**
   * @brief Compact storage of multiple sorted sparse index sets sharing a single allocation.
   *
   * @details A `CompressedSparseIndexRanges<A>` stores `size()` sorted sparse index sets. Each index set
   * is a subset of the integer range [0, range().size()). All index sets share a single
   * contiguous allocation using the narrowest integer type that accommodates the index range
   * (one of uint8, uint16, uint32, or uint64), chosen automatically at construction time.
   *
   * @par Construction
   * Objects are built via one of two builder types and cannot be default-constructed:
   * - `SequencedSparseIndexRangeBuilder`: fill index sets sequentially, one at a time from first to last.
   * - `UnsequencedSparseIndexRangeBuilder`: insert indices across all sets in any order, optionally, declaring sizes first.
   *
   * After construction the storage is compressed: unused capacity is discarded and the index
   * type is narrowed to the smallest type that accommodates the actual maximum index value.
   *
   * @par Access
   * Iterate over all index sets with `begin()`/`end()`, or access individual sets with
   * `operator[]`. Each element is an `IndexRangeView`, a lightweight read-only range over
   * the sorted indices of a single index set.
   *
   * @par Ownership
   * The object is move-only; copy construction and copy assignment are disabled.
   *
   * @tparam A Allocator type (default: `std::allocator<void>`).
   */
  template <class A = std::allocator<void>>
  class CompressedSparseIndexRanges
  {
    using alloc_type = A;
    using alloc_traits = std::allocator_traits<alloc_type>;

    using size_alloc_type = typename alloc_traits::template rebind_alloc<typename alloc_traits::size_type>;
    using size_alloc_traits = std::allocator_traits<size_alloc_type>;

    using IndexVariant = typename Impl::SparseIndexRangeHelper<A>::IndexVariant;

    // Nested type declarations
    class IndexIterator;

    constexpr CompressedSparseIndexRanges() = delete;
    constexpr CompressedSparseIndexRanges(const CompressedSparseIndexRanges &other) = delete;
    constexpr CompressedSparseIndexRanges &operator=(const CompressedSparseIndexRanges &other) = delete;

  public:

    //! Type to access the index sets
    using size_type = typename alloc_traits::size_type;

    //! Type of the indices
    using index_type = uint_fast64_t;

    //! Allocator type used for memory management
    using allocator_type = A;

    //! Iterator over CompressedSparseIndexRanges, iterates over the view of index sets (e.g., rows of the matrix)
    class Iterator;

    //! View of an index set, provides access to the indices of a single index set (e.g., column indices of a single row)
    class IndexRangeView;

    //! Construct from a finished `SequencedSparseIndexRangeBuilder`, taking ownership of its storage and compressing it.
    template<class OtherAlloc>
    requires std::constructible_from<A, OtherAlloc>
    constexpr CompressedSparseIndexRanges(SequencedSparseIndexRangeBuilder<OtherAlloc>&& builder);

    //! Construct from a finished `UnsequencedSparseIndexRangeBuilder`, taking ownership of its storage and compressing it.
    template<class OtherAlloc>
    requires std::constructible_from<A, OtherAlloc>
    constexpr CompressedSparseIndexRanges(UnsequencedSparseIndexRangeBuilder<OtherAlloc>&& builder);

    //! Move-construct, transferring ownership of storage from @p other. @p other is left in a valid but empty state.
    constexpr CompressedSparseIndexRanges(CompressedSparseIndexRanges &&other);

    //! Move-assign, releasing current storage and transferring ownership from @p other. @p other is left in a valid but empty state.
    constexpr CompressedSparseIndexRanges &operator=(CompressedSparseIndexRanges &&other);

    //! Swap contents with @p other honoring allocator swap propagation traits.
    constexpr void swap(CompressedSparseIndexRanges &other);

    //! Swap contents of @p lhs and @p rhs, honoring allocator swap propagation traits.
    friend constexpr void swap(CompressedSparseIndexRanges &lhs, CompressedSparseIndexRanges &rhs)
    {
      lhs.swap(rhs);
    }

    //! Destroy the index sets and release all allocated memory.
    constexpr ~CompressedSparseIndexRanges();

    //! Returns the number of index sets (e.g., the number of rows in a sparse matrix).
    constexpr size_type size() const noexcept;

    //! Returns the total number of indices stored across all index sets (sum of all individual set sizes).
    constexpr size_type count() const;

    //! Returns the integer range [0, n) that every stored index belongs to.
    constexpr IntegralRange<index_type> range() const noexcept;

    //! Returns the flat-storage offset of the start of the i-th index set.
    //! @param i Index-set number; 0 <= i <= size(). Pass i == size() to obtain the total index count.
    constexpr size_type offset(size_type i) const;

    //! Returns a read-only view of the i-th index set.
    //! @param i Index-set number; must satisfy 0 <= i < size().
    constexpr IndexRangeView operator[](size_type i) const { return {this, i}; };

    //! Returns an iterator to the first index set (index set 0).
    constexpr Iterator begin() const { return {this, 0}; }

    //! Returns an iterator past the last index set.
    constexpr Iterator end() const { return {this, size()}; }

    //! Writes all index sets to @p os, one per line in the form "[i]: j0 j1 ...". Returns @p os.
    std::ostream &print(std::ostream &os) const;

    //! Returns true if @p p1 and @p p2 contain identical index sets with identical values.
    constexpr friend bool operator==(const CompressedSparseIndexRanges &p1, const CompressedSparseIndexRanges &p2)
    {
      if (std::addressof(p1) == std::addressof(p2))
        return true;

      if (p1.size_ != p2.size_ || p1.is_size_ != p2.is_size_)
        return false;

      if (!std::equal(p1.offsets_, p1.offsets_ + p1.size_ + 1, p2.offsets_))
        return false;

      return std::visit([&]<typename index_t_1, typename index_t_2>(index_t_1 *indices1, index_t_2 *indices2) -> bool
                        { return std::equal(indices1, indices1 + p1.count(), indices2); }, p1.indices_, p2.indices_);
    }

  private:

    size_type size_; // number of index sets, e.g., rows of the matrix
    index_type is_size_; // size of the index sets, e.g., number of columns of the matrix
    size_type *offsets_; //! offsets to the start of each index set in the flat index storage
    IndexVariant indices_;  //! variant to flat storage of all indices across all index sets
    DUNE_NO_UNIQUE_ADDRESS alloc_type alloc_;
  };

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::Iterator declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Random-access iterator over the index sets of a `CompressedSparseIndexRanges`.
   *
   * @details Dereferencing yields an `IndexRangeView` for the current index set.
   * Obtained via `CompressedSparseIndexRanges::begin()` and `CompressedSparseIndexRanges::end()`.
   * Supports the full random-access iterator interface via `Dune::IteratorFacade`.
   *
   * The iterator is invalidated if the underlying `CompressedSparseIndexRanges` is destroyed or moved.
   */
  template <class A>
  class CompressedSparseIndexRanges<A>::Iterator
      : public Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, IndexRangeView, IndexRangeView, Dune::ProxyArrowResult<IndexRangeView>>
  {
    using Facade = Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, IndexRangeView, IndexRangeView, Dune::ProxyArrowResult<IndexRangeView>>;

    constexpr Iterator(CompressedSparseIndexRanges const *sp, size_type i);

  public:
    using reference = typename Facade::reference;
    using difference_type = typename Facade::difference_type;

    constexpr Iterator() = default;

    constexpr reference operator*() const;

  private:
    friend Dune::IteratorFacadeAccess;
    friend CompressedSparseIndexRanges;

    constexpr const size_type &baseIterator() const;

    constexpr size_type &baseIterator();

    CompressedSparseIndexRanges const *cis_;
    size_type i_;
  };

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::IndexIterator declaration
  // ---------------------------------------------------------------------------

  template <class A>
  class CompressedSparseIndexRanges<A>::IndexIterator
      : public Dune::IteratorFacade<IndexIterator, std::random_access_iterator_tag, index_type, index_type, Dune::ProxyArrowResult<index_type>>
  {
    using Facade = Dune::IteratorFacade<IndexIterator, std::random_access_iterator_tag, index_type, index_type, Dune::ProxyArrowResult<index_type>>;

  public:
    using reference = typename Facade::reference;
    using difference_type = typename Facade::difference_type;

    constexpr IndexIterator(CompressedSparseIndexRanges const *csi, size_type i_, size_type j);

    constexpr IndexIterator() = default;

    constexpr size_type offset() const;

    constexpr reference operator*() const;

  private:
    friend Dune::IteratorFacadeAccess;

    constexpr const size_type &baseIterator() const;

    constexpr size_type &baseIterator();

    CompressedSparseIndexRanges const *cis_;
    size_type i_;
    size_type j_;
  };

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::IndexRangeView declaration
  // ---------------------------------------------------------------------------

  /**
   * @brief Read-only view of a single index set inside a `CompressedSparseIndexRanges`.
   *
   * @details Provides a `std::ranges::view` over the sorted indices of one index set.
   * Obtained via `CompressedSparseIndexRanges::operator[]` or by dereferencing a `CompressedSparseIndexRanges::Iterator`.
   *
   * `IndexRangeView` is a lightweight proxy (pointer + index offset) that is cheap to copy.
   * It becomes invalid if the underlying `CompressedSparseIndexRanges` is destroyed or moved.
   */
  template <class A>
  class CompressedSparseIndexRanges<A>::IndexRangeView
      : public Impl::SparseIndexRangesBase
  {
    friend class CompressedSparseIndexRanges<A>::Iterator;
    friend class CompressedSparseIndexRanges<A>;

    constexpr IndexRangeView(const CompressedSparseIndexRanges *cis, size_type i);

  public:
    IndexRangeView() = default;
    IndexRangeView(const IndexRangeView&) = default;
    IndexRangeView& operator=(const IndexRangeView&) = default;

    //! Type of the indices
    using index_type = typename CompressedSparseIndexRanges<A>::index_type;

    using size_type = typename CompressedSparseIndexRanges<A>::size_type;

    IndexRangeView(IndexRangeView&&) = default;
    IndexRangeView& operator=(IndexRangeView&&) = default;

    //! Returns an iterator to the first index in this index set.
    constexpr IndexIterator begin() const;
    //! Returns an iterator past the last index in this index set.
    constexpr IndexIterator end() const;

    //! Returns the number of indices in this index set.
    constexpr size_type size() const;

    //! Returns true if this index set is empty (i.e., contains no indices).
    constexpr bool empty() const { return size() == 0; }

    //! Returns the first index in this index set. Precondition: the set is not empty.
    constexpr index_type front() const { return *begin(); }

    //! Returns the last index in this index set. Precondition: the set is not empty.
    constexpr index_type back() const { return *(end() - 1); }

    //! Returns the index at local position @p i in this index set. Precondition: i < size().
    constexpr index_type operator[](size_type i) const { return *(begin() + i); }

    //! Returns an integer range [0, n) usable as a local index range over this set.
    constexpr IntegralRange<index_type> range() const;

    //! Returns true if index @p j is contained in this index set. Complexity: O(size()) for small sets, O(log(size())) for large sets.
    constexpr bool contains(index_type j) const;

    //! Returns an iterator to the first index >= @p i in this index set, or `end()` if none exists.
    constexpr IndexIterator lower_bound(index_type i) const;


    constexpr IndexIterator upper_bound(index_type i) const;


    constexpr IndexIterator find(index_type i) const;

    //! Calls @p f(j) for every index @p j in this index set, in ascending order.
    template<std::invocable<index_type> F>
    constexpr void for_each(F &&f) const
    {
      std::visit([&]<typename index_t>(index_t *indices) {
        const auto index_set = std::span(indices + cis_->offset(i_), indices + cis_->offset(i_ + 1));
        std::for_each(std::cbegin(index_set), std::cend(index_set), std::forward<F>(f));
      }, cis_->indices_);
    }

  private:
    CompressedSparseIndexRanges const *cis_;
    size_type i_;
  };

  template<Concepts::BorrowedOrderedIndexRange R>
    requires std::derived_from<R, Impl::SparseIndexRangesBase>
  struct OrderedIndexRangeTraits<R>
  {
    using index_range_type = R;
    using index_type = typename index_range_type::index_type;
    using size_type = typename index_range_type::size_type;

    constexpr static IntegralRange<index_type> range(const index_range_type& index_set)
    {
      return index_set.range();
    }

    constexpr static size_type offset(const index_range_type& index_set, const index_type& value)
    {
      return static_cast<size_type>(std::distance(std::cbegin(index_set), index_set.lower_bound(value)));
    }
  };

  // ---------------------------------------------------------------------------
  // Impl::SparseIndexRangeHelper implementations
  // ---------------------------------------------------------------------------

namespace Impl
{

  template <class A>
  void SparseIndexRangeHelper<A>::debug(std::span<size_type> offsets, IndexVariant indices)
  {
    std::cout << "Index Set Range: [0,  " << offsets.size() - 1 << ")" << std::endl;
    std::cout << "Offsets: ";
    for (auto i : offsets)
      std::cout << i << " ";
    std::cout << "\n";
    std::visit([&]<class index_t>(index_t *indices_ptr) {
      std::cout << "Indices [" << Dune::className<index_t>() << "]: | ";
      for (std::size_t i = 0; i + 1 != offsets.size(); ++i) {
        for (auto j : std::span(indices_ptr + offsets[i], offsets[i + 1] - offsets[i]))
          std::cout << (j == std::numeric_limits<index_t>::max() ? "*" : std::to_string(static_cast<size_type>(j))) << " ";
        std::cout << "| ";
      }
      std::cout << "\n";
    }, indices);
  }


  template <class A>
  constexpr auto SparseIndexRangeHelper<A>::initAllocateOffsets(const allocator_type &allocator, size_type size) -> size_type* {
    size_alloc_type size_alloc(allocator);
    size_type* offsets = size_alloc_traits::allocate(size_alloc, size+1);
    for (size_type i = 0; i != size+1; ++i)
      size_alloc_traits::construct(size_alloc, std::addressof(offsets[i]), size_type{0});
    return offsets;
   }

  template <class A>
  constexpr auto SparseIndexRangeHelper<A>::initAllocateIndices(const allocator_type &allocator, size_type count, size_type max_size) -> std::pair<IndexVariant,size_type> {
    IndexVariant indices;
    // use the smallest possible one for the index set indices array
    size_type max_index = std::max<size_type>(max_size, 1) - 1;
    assert(max_index < std::numeric_limits<uint_least64_t>::max() && "Too many indices for supported index types");
    if (max_index < std::numeric_limits<uint_least8_t>::max())
      indices = IndexVariant{std::in_place_type<uint_least8_t *>, nullptr};
    else if (max_index < std::numeric_limits<uint_least16_t>::max())
      indices = IndexVariant{std::in_place_type<uint_least16_t *>, nullptr};
    else if (max_index < std::numeric_limits<uint_least32_t>::max())
      indices = IndexVariant{std::in_place_type<uint_least32_t *>, nullptr};
    else
      indices = IndexVariant{std::in_place_type<uint_least64_t *>, nullptr};

    // visit assigned pointer
    std::visit([&count, &allocator]<class index_t>(index_t *&indices_ptr) {
      using index_alloc_type = typename alloc_traits::template rebind_alloc<index_t>;
      using index_alloc_traits = std::allocator_traits<index_alloc_type>;
      index_alloc_type index_alloc(allocator);
      // actual allocation of storage
#if __cpp_lib_allocate_at_least >= 202302L
      std::tie(indices_ptr, count) = index_alloc_traits::allocate_at_least(index_alloc, count);
#else
      indices_ptr = index_alloc_traits::allocate(index_alloc, count);
#endif
      // initialize index set indices with invalid value
      for (size_type i = 0; i != count; ++i)
        index_alloc_traits::construct(index_alloc, std::addressof(indices_ptr[i]), std::numeric_limits<index_t>::max());
    }, indices);

    return std::make_pair(indices, count);
  }

  template <class A>
  constexpr void SparseIndexRangeHelper<A>::deallocate(const allocator_type& allocator, IndexVariant indices, size_type* offsets, size_type size) {
    if (offsets)
    {
      std::visit([&]<class index_t>(index_t *indices_ptr) {
        using index_alloc_type = typename alloc_traits::template rebind_alloc<index_t>;
        using index_alloc_traits = std::allocator_traits<index_alloc_type>;
        if (indices_ptr) {
          size_type count = offsets[size];
          index_alloc_type index_alloc(allocator);
          index_alloc_traits::deallocate(index_alloc, indices_ptr, count);
        } }, indices);
      size_alloc_type size_alloc(allocator);
      size_alloc_traits::deallocate(size_alloc, offsets, size + 1);
    }
  }


  template <class A>
  template <class V>
  constexpr V SparseIndexRangeHelper<A>::compress(std::span<size_type> offsets, IndexVariant indices, const allocator_type& alloc)
  {
    // check if there are undefined indices and remove them if necessary
    return std::visit([&alloc, offsets]<class old_index_t>(old_index_t *old_indices) -> V {
      size_type index_count = offsets.back();
      // count the max index and non-used entries
      constexpr old_index_t invalid_index = std::numeric_limits<old_index_t>::max();
      size_type count_invalid = 0;
      size_type max_index = 0;
      auto old_index_set = std::span(old_indices, index_count);
      std::for_each(std::cbegin(old_index_set), std::cend(old_index_set), [&count_invalid, &max_index](old_index_t j) {
        if (j == invalid_index)
          count_invalid++;
        else
          max_index = std::max<size_type>(max_index, j);
      });
      V new_indices = makeSmallestIndexVariant<V>(max_index);

      // nothing to do if there are no undefined indices and the indices already use the selected index type
      if constexpr (std::constructible_from<V, old_index_t*>)
        if (count_invalid == 0 && std::holds_alternative<old_index_t*>(new_indices)) {
          new_indices = V{std::in_place_type<old_index_t *>, old_indices};
          return new_indices;
        }

      // remove undefined indices and update offsets accordingly
      constexpr auto index_defined = [](old_index_t i){ return i != invalid_index; };
      // allocate new arrays for offsets and indices. We use scope guards to ensure that the allocated memory is freed in case of exceptions.
      size_alloc_type size_alloc(alloc);
      size_type *new_offsets = size_alloc_traits::allocate(size_alloc, offsets.size());
      typename Impl::SparseIndexRangeHelper<A>::template DeallocationGuard<size_alloc_type> offsetsScopeGuard{size_alloc, new_offsets, offsets.size()};

      new_indices = std::visit([offsets, old_indices, index_count, count_invalid, index_defined, &new_offsets, &size_alloc]<typename new_index_t>(new_index_t * new_indices_ptr) -> V {
        using new_index_alloc_type = typename alloc_traits::template rebind_alloc<new_index_t>;
        using new_index_alloc_traits = std::allocator_traits<new_index_alloc_type>;
        new_index_alloc_type new_index_alloc(size_alloc);
        new_indices_ptr = new_index_alloc_traits::allocate(new_index_alloc, index_count - count_invalid);
        typename Impl::SparseIndexRangeHelper<A>::template DeallocationGuard<new_index_alloc_type> newIndicesScopeGuard{new_index_alloc, new_indices_ptr, index_count - count_invalid};

        size_alloc_traits::construct(size_alloc, std::addressof(new_offsets[0]), 0);
        for (size_type i = 0; i + 1!= offsets.size(); ++i)
        {
          // count number of defined indices in index set
          auto old_index_set = std::span(old_indices + offsets[i], old_indices + offsets[i + 1]);
          size_type is_count = std::count_if(std::cbegin(old_index_set), std::cend(old_index_set), index_defined);
          // compute new offset for next index set
          size_alloc_traits::construct(size_alloc, std::addressof(new_offsets[i + 1]), new_offsets[i] + is_count);
          // copy defined indices of old index set to new index set
          auto new_index_set = std::span(new_indices_ptr + new_offsets[i], is_count);
          size_type j = 0;
          for (const auto old_idx : old_index_set)
            if (index_defined(old_idx))
              new_index_alloc_traits::construct(new_index_alloc, std::addressof(new_index_set[j++]), static_cast<new_index_t>(old_idx));
        }

        // replace old offsets with new ones
        std::copy_n(new_offsets, offsets.size(), std::begin(offsets));

        // replace old indices with new ones and free old memory
        using old_index_alloc_type = typename alloc_traits::template rebind_alloc<old_index_t>;
        typename Impl::SparseIndexRangeHelper<A>::template DeallocationGuard<old_index_alloc_type> oldIndicesScopeGuard{old_index_alloc_type(size_alloc), old_indices, index_count};
        return {std::exchange(newIndicesScopeGuard.ptr, nullptr)};
      }, new_indices);

      return new_indices;
    }, indices);
  }


  template <class A>
  template<class index_t>
  constexpr bool SparseIndexRangeHelper<A>::addIndex(std::span<index_t> index_set, index_t j, std::invocable auto grow_index_set_fn)
  {
    constexpr index_t invalid_index = std::numeric_limits<index_t>::max();

    // find correct insertion position for new index
    auto it = SparseIndexRangeHelper<A>::lowerBound(index_set, j);

    // check if index is already in index set
    if (it != std::cend(index_set) && *it == j)
      return false;

    // grow number of indices if exhausted
    if (it == std::cend(index_set) || index_set.back() != invalid_index) {
      // save distance to insertion position before growth
      size_type dist = std::distance(std::cbegin(index_set), it);
      // invoke growth function and update index set
      index_set = std::invoke(grow_index_set_fn);
      // check that growth function has made space for new index
      assert(index_set.size() > dist && "Too many entries in index set but the builder cannot grow the indices");
      assert(index_set.back() == invalid_index && "Growth function must initialize new indices with invalid value");
      // restore iterator to correct position after growth
      it = std::begin(index_set) + dist;
    }

    // insert new index and shift other indices if necessary
    do {
      j = std::exchange(*it, j);
      it = (j == invalid_index) ? std::cend(index_set) : std::next(it);
    } while (it != std::cend(index_set));
    return true;
  }

  template<class A>
  constexpr void SparseIndexRangeHelper<A>::growIndices(std::span<size_type> offsets, IndexVariant& indices, size_type new_count, const allocator_type& allocator, auto uninitialized_copy_fn){
    size_type old_count = offsets.back();
    if (new_count <= old_count)
      return;

    // allocate new arrays for indices. We use scope guards to ensure that the allocated memory is freed in case of exceptions.
    std::visit([&indices, &allocator, &uninitialized_copy_fn, offsets, old_count, new_count]<typename index_t>(index_t * old_indices) {
      using index_alloc_type = typename alloc_traits::template rebind_alloc<index_t>;
      using index_alloc_traits = std::allocator_traits<index_alloc_type>;
      using dealloc_guard_type = typename Impl::SparseIndexRangeHelper<A>::template DeallocationGuard<index_alloc_type>;
      index_alloc_type index_alloc(allocator);
      index_t* new_indices = nullptr;
      size_type allocated_count = new_count;
#if __cpp_lib_allocate_at_least >= 202302L
      std::tie(new_indices, allocated_count) = index_alloc_traits::allocate_at_least(index_alloc, allocated_count);
#else
      new_indices = index_alloc_traits::allocate(index_alloc, allocated_count);
#endif
      dealloc_guard_type newIndicesScopeGuard{index_alloc, new_indices, allocated_count};

      // copy values from old indices to new indices
      uninitialized_copy_fn(offsets, std::span<const index_t>(old_indices, old_count), std::span(new_indices, allocated_count));
      // offsets always need to be updated to reflect the new size of the indices array
      assert(offsets.back() == allocated_count);

      // replace old indices with new ones and free old memory
      dealloc_guard_type oldIndicesScopeGuard{index_alloc, old_indices, old_count};
      indices = IndexVariant{std::in_place_type<index_t *>, std::exchange(newIndicesScopeGuard.ptr, nullptr)};
    }, indices);
  }

  template<class A>
  constexpr void SparseIndexRangeHelper<A>::growLastIndexRange(std::span<size_type> offsets, IndexVariant& indices, double growth_factor, const allocator_type& allocator){
    const size_type old_count = offsets.back();
    const size_type new_count = std::max<size_type>(old_count*growth_factor, offsets.size() - 1);
    SparseIndexRangeHelper<A>::growIndices(offsets, indices, new_count, allocator, [allocator]<typename index_t>(std::span<size_type> offsets, std::span<const index_t> old_indices, std::span<index_t> new_indices){
      using index_alloc_type = typename alloc_traits::template rebind_alloc<index_t>;
      using index_alloc_traits = std::allocator_traits<index_alloc_type>;
      index_alloc_type index_alloc(allocator);
      // uninitialized copy of old indices into new indices
      for (size_type j = 0; j != old_indices.size(); ++j)
        index_alloc_traits::construct(index_alloc, std::addressof(new_indices[j]), old_indices[j]);

      // initialize the remaining tail of the new indices with invalid value
      for (size_type j = old_indices.size(); j != new_indices.size(); ++j)
        index_alloc_traits::construct(index_alloc, std::addressof(new_indices[j]), std::numeric_limits<index_t>::max());

      // update offset with new size
      offsets.back() = new_indices.size();
    });
  }

  template<class A>
  constexpr void SparseIndexRangeHelper<A>::growAllIndexRanges(std::span<size_type> offsets, IndexVariant& indices, double target_growth_factor, const allocator_type& allocator){

    if (target_growth_factor <= 1.0 or offsets.size() <= 1)
      return;
    constexpr size_type min_extra_indices = 1;
    size_type max_is_used = 0;
    size_type total_used_indices = 0;
    // compute the total number of used indices across all index sets and the maximum number of used indices in any index set to estimate the new count of indices after growth
    std::visit([offsets, &max_is_used, &total_used_indices]<class index_t>(index_t const * indices_ptr) {
      // grow number of indices for all index sets by target_growth_factor times the current size or by the small index set threshold, whichever is larger
      for (size_type i = 0; i + 1 != offsets.size(); ++i) {
        // get set of indices
        std::span index_set(indices_ptr + offsets[i], offsets[i+1] - offsets[i]);

        // find insertion position for new index to get number of used indices in the index set
        auto it = lowerBound(index_set, std::numeric_limits<index_t>::max());
        size_type used_indices = std::distance(std::cbegin(index_set), it);

        max_is_used = std::max(max_is_used, used_indices);
        total_used_indices += used_indices;
      }
    }, indices);
    size_type target_extra_per_index_set = std::max(static_cast<size_type>(max_is_used*(target_growth_factor - 1.)), min_extra_indices);
    size_type target_extra_indices = (offsets.size() - 1) * target_extra_per_index_set;
    size_type new_count = total_used_indices + target_extra_indices;

    growIndices(offsets, indices, new_count, allocator, [allocator, min_extra_indices, total_used_indices, new_count]<typename index_t>(std::span<size_type> offsets, std::span<const index_t> old_indices, std::span<index_t> new_indices){
      using index_alloc_type = typename alloc_traits::template rebind_alloc<index_t>;
      using index_alloc_traits = std::allocator_traits<index_alloc_type>;
      index_alloc_type index_alloc(allocator);

      // indices may have grown by more than the target extra indices, so we compute an average extra indices per index set
      (void)new_count;
      assert(new_count >= new_indices.size() &&  "Allocated index storage must be at least as large as the new count of indices after growth");

      const size_type extra_per_index_set = (new_indices.size() - total_used_indices) / (offsets.size() - 1);
      (void)min_extra_indices;
      assert(extra_per_index_set >= min_extra_indices && "New index storage must accommodate at least min_extra_indices extra index per used index to ensure progress");

      // carry offset to keep track of the current position in the new indices array
      size_type old_row_begin = 0;
      for (size_type i = 0; i + 1 != offsets.size(); ++i) {
        const size_type old_row_end = offsets[i+1];
        // get set of indices
        std::span old_index_set = old_indices.subspan(old_row_begin, old_row_end - old_row_begin);

        // find insertion position for new index to get number of used indices in the index set
        auto it = SparseIndexRangeHelper<A>::lowerBound(old_index_set, std::numeric_limits<index_t>::max());
        size_type used_indices = std::distance(std::cbegin(old_index_set), it);
        // actual new size of i-th index set
        size_type new_size = used_indices + extra_per_index_set;
        new_size = std::min(new_size, new_indices.size() - offsets[i]); // ensure that we do not exceed the allocated size

        std::span new_index_set = new_indices.subspan(offsets[i], new_size);
        // uninitialized copy of old indices into new indices
        for (size_type j = 0; j != used_indices; ++j)
          index_alloc_traits::construct(index_alloc, std::addressof(new_index_set[j]), old_index_set[j]);
        // initialize the remaining tail of the new index set with invalid value
        for (size_type j = used_indices; j != new_size; ++j)
          index_alloc_traits::construct(index_alloc, std::addressof(new_index_set[j]), std::numeric_limits<index_t>::max());

        // update offset for next index set by the new size of the current index set and move begin of old row to end of old row
        old_row_begin = old_row_end;
        offsets[i+1] = offsets[i] + new_size;
      }

      // due to averaging, it may happen that we do not use all the allocated indices, so we initialize the remaining tail of the new indices with invalid value
      for (size_type j = offsets.back(); j != new_indices.size(); ++j)
        index_alloc_traits::construct(index_alloc, std::addressof(new_indices[j]), std::numeric_limits<index_t>::max());

      // update offset with new size
      offsets.back() = new_indices.size();
    });
  }

  // ---------------------------------------------------------------------------
  // Impl::SparseIndexRangeHelper::DeallocationGuard declaration/implementation
  // ---------------------------------------------------------------------------

  template <class A>
  template <class OtherAlloc>
  struct SparseIndexRangeHelper<A>::DeallocationGuard
  {
    using alloc_traits = std::allocator_traits<OtherAlloc>;

    constexpr DeallocationGuard(OtherAlloc allocator, typename alloc_traits::pointer ptr, typename alloc_traits::size_type size)
        : allocator(allocator), ptr(ptr), size(size)
    {
    }

    DeallocationGuard(const DeallocationGuard &) = delete;
    DeallocationGuard &operator=(const DeallocationGuard &) = delete;

    constexpr ~DeallocationGuard()
    {
      if (ptr)
        alloc_traits::deallocate(allocator, ptr, size);
    }

    OtherAlloc allocator;
    typename alloc_traits::pointer ptr;
    typename alloc_traits::size_type size;
  };

} // namespace Impl


  // ---------------------------------------------------------------------------
  // SparseIndexRanges implementations
  // ---------------------------------------------------------------------------

  template <class I, class A>
  template<class OtherAlloc>
  requires std::constructible_from<A, OtherAlloc>
  constexpr SparseIndexRanges<I,A>::SparseIndexRanges(SequencedSparseIndexRangeBuilder<OtherAlloc>&& builder)
    : size_(builder.size_),
      is_size_(builder.is_size_),
      offsets_(nullptr),
      indices_(nullptr),
      alloc_(builder.alloc_)
  {
    assert(builder.stage_ == SequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Built && "Sparsity pattern is not in correct build mode");

    // Compress on builder-owned storage
    using IndexVariant = std::variant<I*>;
    IndexVariant indices_var = Impl::SparseIndexRangeHelper<A>::template compress<IndexVariant>(std::span(builder.offsets_, size_ + 1), builder.indices_, alloc_);

    // Transfer ownership only after compression succeeds
    offsets_ = std::exchange(builder.offsets_, nullptr);
    indices_ = std::get<I*>(std::move(indices_var));
    size_ = std::exchange(builder.size_, 0);
    is_size_ = std::exchange(builder.is_size_, 0);

    builder.stage_ = SequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Invalid;
  }

  template <class I, class A>
  template<class OtherAlloc>
  requires std::constructible_from<A, OtherAlloc>
  constexpr SparseIndexRanges<I,A>::SparseIndexRanges(UnsequencedSparseIndexRangeBuilder<OtherAlloc>&& builder)
    : size_(builder.size_),
      is_size_(builder.is_size_),
      offsets_(nullptr),
      indices_(nullptr),
      alloc_(builder.alloc_)
  {
    assert(builder.stage_ == UnsequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::BuildSetIndices && "Sparsity pattern is not in correct build mode");

    // Compress on builder-owned storage
    using IndexVariant = std::variant<I*>;
    IndexVariant indices_var = Impl::SparseIndexRangeHelper<A>::template compress<IndexVariant>(std::span(builder.offsets_, size_ + 1), builder.indices_, alloc_);

    // Transfer ownership only after compression succeeds
    offsets_ = std::exchange(builder.offsets_, nullptr);
    indices_ = std::get<I*>(std::move(indices_var));
    size_ = std::exchange(builder.size_, 0);
    is_size_ = std::exchange(builder.is_size_, 0);

    builder.stage_ = UnsequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Invalid;
  }

  template <class I, class A>
  constexpr SparseIndexRanges<I,A>::SparseIndexRanges(SparseIndexRanges &&other)
      : size_(std::exchange(other.size_, 0)),
        is_size_(std::exchange(other.is_size_, 0)),
        offsets_(std::exchange(other.offsets_, nullptr)),
        indices_(std::exchange(other.indices_, nullptr)),
        alloc_(std::move(other.alloc_))
  {}

  template <class I, class A>
  constexpr SparseIndexRanges<I,A> &SparseIndexRanges<I,A>::operator=(SparseIndexRanges &&other)
  {
    if (this == &other)
      return *this;

    deallocate(alloc_, indices_, offsets_, size_);

    is_size_ = std::exchange(other.is_size_, 0);
    size_ = std::exchange(other.size_, 0);
    offsets_ = std::exchange(other.offsets_, nullptr);
    indices_ = std::exchange(other.indices_, nullptr);
    alloc_ = std::move(other.alloc_);
    return *this;
  }

  template <class I, class A>
  constexpr void SparseIndexRanges<I,A>::swap(SparseIndexRanges &other)
  {
    if (this == &other)
      return;

    if constexpr (!std::allocator_traits<allocator_type>::propagate_on_container_swap::value
                  && !std::allocator_traits<allocator_type>::is_always_equal::value)
      if (alloc_ != other.alloc_)
        DUNE_THROW(InvalidStateException,
                   "Cannot swap SparseIndexRanges with unequal allocators when propagate_on_container_swap is false");

    using std::swap;
    swap(size_, other.size_);
    swap(is_size_, other.is_size_);
    swap(offsets_, other.offsets_);
    swap(indices_, other.indices_);
    if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value)
      swap(alloc_, other.alloc_);
  }

  template <class I, class A>
  constexpr SparseIndexRanges<I,A>::~SparseIndexRanges()
  {
    Impl::SparseIndexRangeHelper<A>::deallocate(alloc_, indices_, offsets_, size_);
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::size() const noexcept -> size_type
  {
    return size_;
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::range() const noexcept -> IntegralRange<index_type>
  {
    return {0, is_size_};
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::count() const -> size_type
  {
    return offset(size_);
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::offset(size_type i) const -> size_type
  {
    assert(offsets_);
    return std::span(offsets_, size_ + 1)[i];
  }

  template <class I, class A>
  std::ostream &SparseIndexRanges<I,A>::print(std::ostream &os) const
  {
    for (size_type i = 0; i != size_; ++i)
    {
      os << "[" << i << "]: ";
      std::span index_set(indices_ + offsets_[i], offsets_[i + 1] - offsets_[i]);
      for (auto idx : index_set)
        os << static_cast<size_type>(idx) << " ";
      os << '\n';
    };
    return os;
  }

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::Iterator implementation
  // ---------------------------------------------------------------------------

  template <class I, class A>
  constexpr SparseIndexRanges<I,A>::Iterator::Iterator(SparseIndexRanges const *sp, size_type i)
      : cis_(sp), i_(i)
  {}

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::Iterator::operator*() const -> reference
  {
    return IndexRangeView(cis_, i_);
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::Iterator::baseIterator() const -> const size_type &
  {
    return i_;
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::Iterator::baseIterator() -> size_type &
  {
    return i_;
  }

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::IndexIterator implementation
  // ---------------------------------------------------------------------------

  template <class I, class A>
  constexpr SparseIndexRanges<I,A>::IndexIterator::IndexIterator(SparseIndexRanges const *csi, size_type i, size_type j)
      : cis_(csi), i_(i), j_(j)
  {}

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexIterator::offset() const -> size_type
  {
    return cis_->offsets_[i_] + j_;
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexIterator::operator*() const -> reference
  {
    return cis_->indices_[offset()];
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexIterator::baseIterator() const -> const size_type &
  {
    return j_;
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexIterator::baseIterator() -> size_type &
  {
    return j_;
  }

  // ---------------------------------------------------------------------------
  // SparseIndexRanges::IndexRangeView implementation
  // ---------------------------------------------------------------------------

  template <class I, class A>
  constexpr SparseIndexRanges<I,A>::IndexRangeView::IndexRangeView(const SparseIndexRanges *cis, size_type i)
      : cis_(cis), i_(i)
  {}

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::begin() const -> IndexIterator
  {
    return {cis_, i_, 0};
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::end() const -> IndexIterator
  {
    return {cis_, i_, size()};
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::size() const -> size_type
  {
    return cis_->offsets_[i_ + 1] - cis_->offsets_[i_];
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::range() const -> IntegralRange<index_type>
  {
    return {0, cis_->is_size_};
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::contains(index_type j) const -> bool
  {
    auto index_set = std::span(cis_->indices_ + cis_->offsets_[i_], size());
    auto lb = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, j);
    return lb != std::cend(index_set);
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::lower_bound(index_type j) const -> IndexIterator
  {
    std::span index_set(cis_->indices_ + begin().offset(), end().offset() - begin().offset());
    auto it = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, j);
    return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
    return {};
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::upper_bound(index_type j) const -> IndexIterator
  {
    std::span index_set(cis_->indices_ + begin().offset(), end().offset() - begin().offset());
    auto it = Impl::SparseIndexRangeHelper<A>::upperBound(index_set, j);
    return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
  }

  template <class I, class A>
  constexpr auto SparseIndexRanges<I,A>::IndexRangeView::find(index_type j) const -> IndexIterator
  {
    std::span index_set(cis_->indices_ + begin().offset(), end().offset() - begin().offset());
    auto it = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, j);
    if (it != std::cend(index_set) && *it == j)
      return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
    else
      return end();
  }

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges implementations
  // ---------------------------------------------------------------------------

  template <class A>
  template<class OtherAlloc>
  requires std::constructible_from<A, OtherAlloc>
  constexpr CompressedSparseIndexRanges<A>::CompressedSparseIndexRanges(SequencedSparseIndexRangeBuilder<OtherAlloc> &&builder)
      : size_(builder.size_),
        is_size_(builder.is_size_),
        offsets_(nullptr),
        indices_(static_cast<uint_least64_t *>(nullptr)),
        alloc_(builder.alloc_)
  {
    assert(builder.stage_ == SequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Built && "Sparsity pattern is not in correct build mode");

    // Compress on builder-owned storage
    IndexVariant compressed_indices = Impl::SparseIndexRangeHelper<A>::compress(std::span<size_type>(builder.offsets_, size_ + 1), builder.indices_, alloc_);

    // Transfer ownership only after compression succeeds
    offsets_ = std::exchange(builder.offsets_, nullptr);
    indices_ = std::move(compressed_indices);
    size_ = std::exchange(builder.size_, 0);
    is_size_ = std::exchange(builder.is_size_, 0);

    builder.stage_ = SequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Invalid;
  }

  template <class A>
  template<class OtherAlloc>
  requires std::constructible_from<A, OtherAlloc>
  constexpr CompressedSparseIndexRanges<A>::CompressedSparseIndexRanges(UnsequencedSparseIndexRangeBuilder<OtherAlloc>&& builder)
      : size_(builder.size_),
        is_size_(builder.is_size_),
        offsets_(nullptr),
        indices_(static_cast<uint_least64_t *>(nullptr)),
        alloc_(builder.alloc_)
  {
    assert(builder.stage_ == UnsequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::BuildSetIndices && "Sparsity pattern is not in correct build mode");

    // Compress on builder-owned storage
    IndexVariant compressed_indices = Impl::SparseIndexRangeHelper<A>::compress(std::span<size_type>(builder.offsets_, size_ + 1), builder.indices_, alloc_);

    // Transfer ownership only after compression succeeds
    offsets_ = std::exchange(builder.offsets_, nullptr);
    indices_ = std::move(compressed_indices);
    size_ = std::exchange(builder.size_, 0);
    is_size_ = std::exchange(builder.is_size_, 0);

    builder.stage_ = UnsequencedSparseIndexRangeBuilder<OtherAlloc>::Stage::Invalid;
  }

  template <class A>
  constexpr CompressedSparseIndexRanges<A>::CompressedSparseIndexRanges(CompressedSparseIndexRanges &&other)
      : size_(std::exchange(other.size_, 0)),
        is_size_(std::exchange(other.is_size_, 0)),
        offsets_(std::exchange(other.offsets_, nullptr)),
        indices_(std::move(other.indices_)),
        alloc_(std::move(other.alloc_))
  {
    other.indices_ = IndexVariant{std::in_place_type<uint_least64_t *>, nullptr};
  }

  template <class A>
  constexpr CompressedSparseIndexRanges<A> &CompressedSparseIndexRanges<A>::operator=(CompressedSparseIndexRanges &&other)
  {
    if (this == &other)
      return *this;

    deallocate(alloc_, indices_, offsets_, size_);

    is_size_ = std::exchange(other.is_size_, 0);
    size_ = std::exchange(other.size_, 0);
    offsets_ = std::exchange(other.offsets_, nullptr);
    indices_ = std::move(other.indices_);
    other.indices_ = IndexVariant{std::in_place_type<uint_least64_t *>, nullptr};
    alloc_ = std::move(other.alloc_);
    return *this;
  }

  template <class A>
  constexpr void CompressedSparseIndexRanges<A>::swap(CompressedSparseIndexRanges &other)
  {
    if (this == &other)
      return;

    if constexpr (!std::allocator_traits<allocator_type>::propagate_on_container_swap::value
                  && !std::allocator_traits<allocator_type>::is_always_equal::value)
      if (alloc_ != other.alloc_)
        DUNE_THROW(InvalidStateException,
                   "Cannot swap CompressedSparseIndexRanges with unequal allocators when propagate_on_container_swap is false");

    using std::swap;
    swap(size_, other.size_);
    swap(is_size_, other.is_size_);
    swap(offsets_, other.offsets_);
    swap(indices_, other.indices_);
    if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value)
      swap(alloc_, other.alloc_);
  }

  template <class A>
  constexpr CompressedSparseIndexRanges<A>::~CompressedSparseIndexRanges()
  {
    Impl::SparseIndexRangeHelper<A>::deallocate(alloc_, indices_, offsets_, size_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::size() const noexcept -> size_type
  {
    return size_;
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::range() const noexcept -> IntegralRange<index_type>
  {
    return {0, is_size_};
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::count() const -> size_type
  {
    return offset(size_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::offset(size_type i) const -> size_type
  {
    assert(offsets_);
    assert(i <= size_ && "Offset index out of bounds");
    return *(offsets_ + i);
  }

  template <class A>
  std::ostream &CompressedSparseIndexRanges<A>::print(std::ostream &os) const
  {
    std::visit([&]<typename index_t>(index_t *indices)
    {
      for (size_type i = 0; i != size_; ++i)
      {
        os << "[" << i << "]: ";
        std::span index_set(indices + offsets_[i], offsets_[i + 1] - offsets_[i]);
        for (auto idx : index_set)
          os << static_cast<size_type>(idx) << " ";
        os << '\n';
      } }, indices_);
    return os;
  }

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::Iterator implementation
  // ---------------------------------------------------------------------------

  template <class A>
  constexpr CompressedSparseIndexRanges<A>::Iterator::Iterator(CompressedSparseIndexRanges const *sp, size_type i)
      : cis_(sp), i_(i)
  {}

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::Iterator::operator*() const -> reference
  {
    return IndexRangeView(cis_, i_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::Iterator::baseIterator() const -> const size_type &
  {
    return i_;
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::Iterator::baseIterator() -> size_type &
  {
    return i_;
  }

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::IndexIterator implementation
  // ---------------------------------------------------------------------------

  template <class A>
  constexpr CompressedSparseIndexRanges<A>::IndexIterator::IndexIterator(CompressedSparseIndexRanges const *csi, size_type i, size_type j)
      : cis_(csi), i_(i), j_(j)
  {}

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexIterator::offset() const -> size_type
  {
    return cis_->offsets_[i_] + j_;
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexIterator::operator*() const -> reference
  {
    return std::visit([o = offset()](auto indices) -> index_type {
      return static_cast<index_type>(indices[o]);
    }, cis_->indices_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexIterator::baseIterator() const -> const size_type &
  {
    return j_;
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexIterator::baseIterator() -> size_type &
  {
    return j_;
  }

  // ---------------------------------------------------------------------------
  // CompressedSparseIndexRanges::IndexRangeView implementation
  // ---------------------------------------------------------------------------

  template <class A>
  constexpr CompressedSparseIndexRanges<A>::IndexRangeView::IndexRangeView(const CompressedSparseIndexRanges *cis, size_type i)
      : cis_(cis), i_(i)
  {}

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::begin() const -> IndexIterator
  {
    return {cis_, i_, 0};
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::end() const -> IndexIterator
  {
    return {cis_, i_, size()};
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::size() const -> size_type
  {
    return cis_->offsets_[i_ + 1] - cis_->offsets_[i_];
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::range() const -> IntegralRange<index_type>
  {
    return {0, cis_->is_size_};
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::contains(index_type j) const -> bool
  {
    return std::visit([this, j]<typename index_t>(index_t *indices) -> bool {
      auto index_set = std::span(indices + cis_->offsets_[i_], size());
      auto lb = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, j);
      return lb != std::cend(index_set);
    }, cis_->indices_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::lower_bound(index_type j) const -> IndexIterator
  {
    return std::visit([&]<typename index_t>(index_t *indices) {
      std::span<const index_t> index_set(indices + begin().offset(), end().offset() - begin().offset());
      auto it = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, static_cast<index_t>(j));
      return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
    }, cis_->indices_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::upper_bound(index_type j) const -> IndexIterator
  {
    return std::visit([&]<typename index_t>(index_t *indices) {
      std::span<const index_t> index_set(indices + begin().offset(), end().offset() - begin().offset());
      auto it = Impl::SparseIndexRangeHelper<A>::upperBound(index_set, static_cast<index_t>(j));
      return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
    }, cis_->indices_);
  }

  template <class A>
  constexpr auto CompressedSparseIndexRanges<A>::IndexRangeView::find(index_type j) const -> IndexIterator
  {
    return std::visit([&]<typename index_t>(index_t *indices) {
      std::span<const index_t> index_set(indices + begin().offset(), end().offset() - begin().offset());
      auto it = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, static_cast<index_t>(j));
      if (it != std::cend(index_set) && *it == static_cast<index_t>(j))
        return IndexIterator(cis_, i_, std::distance(std::cbegin(index_set), it));
      else
        return end();
    }, cis_->indices_);
  }

  // ---------------------------------------------------------------------------
  // UnsequencedSparseIndexRangeBuilder declaration
  // ---------------------------------------------------------------------------

  template <class A>
  constexpr UnsequencedSparseIndexRangeBuilder<A>::UnsequencedSparseIndexRangeBuilder(size_type size, size_type is_size, size_type avg_indices, allocator_type allocator)
      : size_(size),
        is_size_(is_size),
        offsets_(nullptr),
        indices_(static_cast<uint_least64_t *>(nullptr)),
        alloc_(allocator),
        stage_(Stage::BuildSizes)
  {
    offsets_ = Impl::SparseIndexRangeHelper<A>::initAllocateOffsets(alloc_, size_);
    std::span offsets(offsets_, size_ + 1);
    // initialize offsets with equal distribution
    for (size_type i = 0; i + 1 < offsets.size(); ++i)
      offsets[i] = avg_indices;
    offsets.back() = 0;
  }

  template <class A>
  constexpr UnsequencedSparseIndexRangeBuilder<A>::~UnsequencedSparseIndexRangeBuilder()
  {
    Impl::SparseIndexRangeHelper<A>::deallocate(alloc_, indices_, offsets_, size_);
  }

  template <class A>
  constexpr void UnsequencedSparseIndexRangeBuilder<A>::setSize(size_type i, size_type s)
  {
    assert(stage_ == Stage::BuildSizes && "Sparsity pattern is not in correct build mode");
    offsets_[i] = s;
  }

  template <class A>
  constexpr void UnsequencedSparseIndexRangeBuilder<A>::incrementSize(size_type i, size_type increment)
  {
    assert(stage_ == Stage::BuildSizes && "Sparsity pattern is not in correct build mode");
    offsets_[i] += increment;
  }

  template <class A>
  constexpr void UnsequencedSparseIndexRangeBuilder<A>::endSizes()
  {
    assert(stage_ == Stage::BuildSizes && "Sparsity pattern is not in correct build mode");
    stage_ = Stage::BuildSetIndices;
    // compute offsets from index set sizes
    std::exclusive_scan(offsets_, offsets_ + size_ + 1, offsets_, size_type{0});

    std::visit([](auto ptr){
      assert(ptr == nullptr && "Indices pointer must be null before initialization");
    }, indices_);
    std::tie(indices_, offsets_[size_]) = Impl::SparseIndexRangeHelper<A>::initAllocateIndices(alloc_, offsets_[size_], is_size_);
  }

  template <class A>
  constexpr void UnsequencedSparseIndexRangeBuilder<A>::addIndex(size_type i, index_type j)
  {
    assert(i < size_ && "There are less index sets than expected for given index i");
    assert(j < is_size_ && "Out of bounds of index set range");

    if (stage_ == Stage::BuildSizes)
      endSizes();

    assert(stage_ == Stage::BuildSetIndices && "Sparsity pattern is not in correct build mode");

    assert(offsets_[i + 1] >= offsets_[i] && "Invalid offsets");
    auto offsets = std::span(offsets_, size_ + 1);

    std::visit([this, j, i, offsets]<class index_t>(index_t * indices) {
      index_t jj = static_cast<index_t>(j);
      Impl::SparseIndexRangeHelper<A>::addIndex(std::span(indices + offsets[i], indices + offsets[i + 1]), jj, [this, i, offsets]() -> std::span<index_t> {
        double growth_factor = 2.5;
        Impl::SparseIndexRangeHelper<A>::growAllIndexRanges(offsets, indices_, growth_factor, alloc_);
        index_t * indices = std::get<index_t *>(indices_);
        return std::span(indices + offsets[i], indices + offsets[i + 1]);
      });
    }, indices_);
  }

  // ---------------------------------------------------------------------------
  // SequencedSparseIndexRangeBuilder implementation
  // ---------------------------------------------------------------------------

  template<class A>
  constexpr SequencedSparseIndexRangeBuilder<A>::SequencedSparseIndexRangeBuilder(size_type size, size_type is_size, size_type avg_indices, allocator_type allocator)
    : size_{size}
    , is_size_{is_size}
    , alloc_{allocator}
  {
    offsets_ = Impl::SparseIndexRangeHelper<A>::initAllocateOffsets(alloc_, size_);
    std::tie(indices_, offsets_[size_]) = Impl::SparseIndexRangeHelper<A>::initAllocateIndices(allocator, avg_indices * size_, is_size_);
    stage_ = Stage::Allocated;
  }

  template<class A>
  constexpr SequencedSparseIndexRangeBuilder<A>::~SequencedSparseIndexRangeBuilder()
  {
    Impl::SparseIndexRangeHelper<A>::deallocate(alloc_, indices_, offsets_, size_);
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::begin() -> IndexRangeBuilder
  {
    return {this};
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::end() -> EndSentinel
  {
    return {};
  }

  template<class A>
  constexpr SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::IndexRangeBuilder(SequencedSparseIndexRangeBuilder* builder)
    : builder_{builder}
    , i_{0}
    , is_count_{0}
    , stage_{Stage::BuildingIndexRanges}
  {
    assert(builder_->stage_ == Stage::Allocated && "IndexRangeBuilder can only be created once");
    builder_->stage_ = stage_;
    builder_->offsets_[i_ + 1] = size_type(builder_->offsets_[builder_->size_] > 0);
  }

  template<class A>
  constexpr SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::IndexRangeBuilder(IndexRangeBuilder && other) {
    *this = std::move(other);
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::operator=(IndexRangeBuilder && other) -> IndexRangeBuilder& {
    builder_ = std::exchange(other.builder_, nullptr);
    i_ = std::exchange(other.i_, 0);
    is_count_ = std::exchange(other.is_count_, 0);
    stage_ = std::exchange(other.stage_, Stage::Invalid);
    return *this;
  }

  template<class A>
  constexpr void SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::insert(index_type j) {
    assert(builder_->stage_ == stage_);
    assert(i_ < builder_->size_ && "There are less index sets than expected for given index i");
    assert(j < builder_->is_size_ && "Out of bounds of index set range");
    assert(*this != EndSentinel() && "The IndexRangeBuilder is past the last index set");

    std::span<size_type> offsets(builder_->offsets_, builder_->size_ + 1);

    assert(offsets[i_ + 1] >= offsets[i_] && "Invalid offsets");

    // add index j to i-th index set. Grow offsets if needed.
    bool new_index_added = std::visit([offsets, j, i = i_, builder = builder_]<class index_t>(index_t * indices) {
      index_t jj = static_cast<index_t>(j);
      return Impl::SparseIndexRangeHelper<A>::addIndex(std::span(indices + offsets[i], indices + offsets[i + 1]), jj, [i, offsets, builder]() -> std::span<index_t> {
        double growth_factor = 2.0;
        const size_type old_size = offsets[i + 1] - offsets[i];
        // check if there is space left in the tail of the indices, if not, grow the indices by expanding the last index set
        if (offsets.back() - offsets[i + 1] == 0)
          Impl::SparseIndexRangeHelper<A>::growLastIndexRange(offsets, builder->indices_, growth_factor, builder->alloc_);
        // set new size of current index set by growth_factor times the current size or by the small index set threshold, whichever is larger
        size_type target_new_size = std::max<size_type>(std::max<size_type>(1, old_size * growth_factor), Impl::SPARSE_SEARCH_THRESHOLD/sizeof(index_type));
        // grow index set up to the current buffer boundary
        offsets[i + 1] = std::min(offsets[i + 1] + target_new_size, offsets.back());
        index_t* indices = std::get<index_t *>(builder->indices_);
        auto index_set = std::span(indices + offsets[i], indices + offsets[i + 1]);
        assert(index_set.size() > old_size && "Invalid offsets after growth");
        return index_set;
      });
    }, builder_->indices_);

    if (new_index_added)
      ++is_count_;
  }

  template<class A>
  template<std::forward_iterator It, std::sentinel_for<It> End>
  constexpr void SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::overrideIndices(It begin,
                                                                                          End end,
                                                                                          bool do_sort,
                                                                                          bool do_unique)
  {
    assert(builder_->stage_ == stage_);
    assert(i_ < builder_->size_ && "There are less index sets than expected for given index i");
    assert(*this != EndSentinel() && "The IndexRangeBuilder is past the last index set");

    std::span<size_type> offsets(builder_->offsets_, builder_->size_ + 1);
    assert(offsets[i_ + 1] >= offsets[i_] && "Invalid offsets");

    const size_type row_begin = offsets[i_];
    const size_type requested_count = static_cast<size_type>(std::distance(begin, end));
    const size_type available_count = offsets.back() - row_begin;

    if (available_count < requested_count) {
      const double growth_factor =
      offsets.back() > 0
          ? static_cast<double>(offsets.back() + (requested_count - available_count) + 1) / static_cast<double>(offsets.back())
          : 2.0;
      Impl::SparseIndexRangeHelper<A>::growLastIndexRange(offsets,
                                                          builder_->indices_,
                                                          growth_factor,
                                                          builder_->alloc_);
    }

    assert(offsets.back() - row_begin >= requested_count && "Growing index set failed to provide enough space for new indices");

    // Extend the active buffer for this row when the replacement needs more room.
    offsets[i_ + 1] = std::max(offsets[i_ + 1], row_begin + requested_count);

    size_type final_count = requested_count;
    std::visit([&, row_begin]<class index_t>(index_t * indices) {
      constexpr index_t invalid_index = std::numeric_limits<index_t>::max();
      auto row = std::span(indices + row_begin, offsets[i_ + 1] - row_begin);

      auto out = std::begin(row);
      for (auto it = begin; it != end; ++it, ++out) {
        assert(*it < builder_->is_size_ && "Out of bounds of index set range");
        *out = static_cast<index_t>(*it);
      }

      if (do_sort)
        std::sort(std::begin(row), out);
      if (do_unique)
        out = std::unique(std::begin(row), out);

      std::fill(out, std::end(row), invalid_index);
      final_count = static_cast<size_type>(std::distance(std::cbegin(row), out));
    }, builder_->indices_);

    is_count_ = final_count;
  }

  template<class A>
  constexpr bool SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::contains(index_type j) const {
    assert(builder_->stage_ == stage_);
    assert(i_ < builder_->size_ && "There are less index sets than expected for given index i");
    assert(j < builder_->is_size_ && "Out of bounds of index set range");
    assert(*this != EndSentinel() && "The IndexRangeBuilder is past the last index set");
    size_type offsets_begin = builder_->offsets_[i_];
    size_type offsets_end = builder_->offsets_[i_ + 1];
    assert(offsets_end >= offsets_begin && "Invalid offsets");

    return std::visit([&]<typename index_t>(const index_t *indices) {
      // get set of indices
      std::span index_set(indices + offsets_begin, offsets_end - offsets_begin);
      const index_t jj = static_cast<index_t>(j);

      // find correct insertion position for new index
      auto it = Impl::SparseIndexRangeHelper<A>::lowerBound(index_set, jj);

      // check if index is in the set
      return it != std::cend(index_set) && *it == jj;
    }, builder_->indices_);
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::index() const -> size_type {
    return i_;
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::operator++() -> IndexRangeBuilder& {
    assert(builder_ && "IndexRangeBuilder is not valid");
    assert(builder_->stage_ == Stage::BuildingIndexRanges);
    assert(*this != EndSentinel() && "The IndexRangeBuilder is past the last index set");

    auto offsets = std::span(builder_->offsets_, builder_->size_ + 1);
    size_type current_offset = offsets[i_];
    ++i_;
    // set next index set to have a small index set or less
    if (i_ < builder_->size_) {
      const std::size_t index_size = std::visit([]<class index_t>(index_t *) { return sizeof(index_t); }, builder_->indices_);
      // set up next index set to have SPARSE_SEARCH_THRESHOLD bytes of indices or less
      offsets[i_] = current_offset + std::exchange(is_count_, 0);
      offsets[i_ + 1] = std::min(offsets[i_] + std::max<size_type>(Impl::SPARSE_SEARCH_THRESHOLD/index_size, 1), offsets[builder_->size_]);
    } else {
      stage_ = builder_->stage_ = Stage::Built;
    }
    return *this;
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::size() const -> size_type {
    assert(*this != EndSentinel() && "The IndexRangeBuilder is past the last index set");
    return is_count_;
  }

  template<class A>
  constexpr auto SequencedSparseIndexRangeBuilder<A>::IndexRangeBuilder::operator*() -> IndexRangeBuilder& {
    return *this;
  }

} // namespace Dune::Experimental

// Opt-in to std::ranges::borrowed_range for any type derived from Impl::SparseIndexRangesBase.
template <std::derived_from<Dune::Experimental::Impl::SparseIndexRangesBase> T>
constexpr bool std::ranges::enable_borrowed_range<T> = true;

#endif // DUNE_ISTL_EXPERIMENTAL_SPARSEINDEXRANGES_HH
