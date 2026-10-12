// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_EXPERIMENTAL_BCRSMATRIX_HH
#define DUNE_ISTL_EXPERIMENTAL_BCRSMATRIX_HH

#include "matrix.hh"
#include "sparseindexranges.hh"

#include <dune/common/scalarmatrixview.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/overloadset.hh>
#include <dune/common/std/no_unique_address.hh>
#include <dune/common/std/type_traits.hh>

#include <memory>
#include <ranges>

namespace Dune::Experimental {
/**
 * \brief Block compressed row storage matrix.
 *
 * BCRSMatrix is an sparse matrix inheriting from the Matrix class
 * explicitly instantiated with SparseIndexRanges as the pattern type.
 * It is designed to be a modernized, allocator-aware replacement for BCRSMatrix
 * that preserves the familiar matrix interface and build workflow while improving
 * performance and control over construction and ownership.
 *
 * \par What stays familiar compared to BCRSMatrix
 * - Constructors with same syntax and similar semantics.
 * - Row-wise and random/implicit build modes.
 * - Matrix/vector vector operations are compatible with existing DUNE
 *   algorithms and tests.
 *
 * \par What is different from BCRSMatrix
 * - There is an extra template parameter for the index type used to store the
 *   column indices whose default type is uint_least32_t which saves
 *   around 25% of memory.
 * - The matrix is an allocator aware container, usable with statefull allocators.
 * - The matrix is now movable without copying internal data.
 * - There is no build stage related structures or methods.
 * - The rows are now proxy objects created on demand which means that mutable
 *   references to rows objects are not possible anymore.
 * - Mutable and constant row iterators are now different.
 * - The pattern is internally built with with the SparseIndexRanges builders.
 *   While it offers the same syntax, there are a few differences to be aware of:
 *   - The build mode switches to unknown after the pattern construction is finalized.
 *     Thus, if the user needs to fill the matrix in multiple stages, they need to
 *     select the build mode again.
 *   - Pattern is entirely decoupled from the matrix, allowing to share
 *     the pattern on other types, e.g., for a matrix in lower precision.
 *   - The entire pattern, including sizes and row pointers, is shared when
 *     copying the matrix.
 *   - Implicit and row-wise build modes are essentially the same under the
 *     hood.
 *   - All build modes grow the pattern dynamically and are compressed once
 *     finalized.
 *   - The implicit build mode does not support storing blocks during the
 *     pattern construction. In order to support the same syntax as the BCRSMatrix
 *     implicit build mode, the entry(i, j) method that only accepts assignment of
 *     values that equal to default constructed blocks.
 *   - The setSize() method should be used after selecting the build mode and
 *     before the pattern construction is finalized.
 *   - The N() and M() methods are not available during the pattern
 *     construction.
 *   - The type of the createend() iterator is a sentinel comparable with the
 *     corresponding createbegin() iterator, but their type is not the same.
 *
 * \par Build modes
 * - BuildMode::row_wise: fill rows sequentially using createbegin()/createend().
 * - BuildMode::random / BuildMode::implicit: insert indices in arbitrary order,
 *   optionally pre-declaring row sizes.
 * - BuildMode::unknown: no build mode selected or construction has completed.
 *
 * \todo Implement setIndicesNoSort and setIndicesSorted methods for row-wise build mode.
 */

template<class B, class I = uint_least32_t, class A = std::allocator<B>>
class BCRSMatrix : public Matrix<B, SparseIndexRanges<I>, A>
{
  //! Allocator type for the block storage.
  using block_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<B>;
  //! Allocator traits for the block storage allocator.
  using block_alloc_traits = std::allocator_traits<block_alloc_type>;
  //! Allocator type for the pattern storage.
  using void_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<void>;

  //! Type of the sequenced builder for the internal pattern construction.
  using SequencedPatternBuilder = SequencedSparseIndexRangeBuilder<void_alloc_type>;
  //! Type of the unsequenced builder for the internal pattern construction.
  using UnsequencedPatternBuilder = UnsequencedSparseIndexRangeBuilder<void_alloc_type>;

  //! Custom deleter for the builder unique pointers.
  struct AllocDeleter
  {
    template<class T>
    void operator()(T* builder)
    {
      using rebound_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<T>;
      rebound_alloc_type rebound_alloc(alloc_);
      using rebound_alloc_traits = typename std::allocator_traits<rebound_alloc_type>;
      rebound_alloc_traits::destroy(rebound_alloc, builder);
      rebound_alloc_traits::deallocate(rebound_alloc, builder, 1);
    }
    DUNE_NO_UNIQUE_ADDRESS A alloc_;
  };

  //! Storage of the active builder during pattern construction.
  using BuilderVariant = std::variant<
    std::monostate,
    std::unique_ptr<SequencedPatternBuilder, AllocDeleter>,
    std::unique_ptr<UnsequencedPatternBuilder, AllocDeleter>>;

  /** \brief A helper for handling default block assignments during implicit build.
   *
   * This class is returned by the entry(i, j) method during implicit build to
   * maintain the same syntax as the BCRSMatrix implicit build mode.
   * However, it only allows assignment of values that are equal to the default
   * constructed block or field type, and does not actually store any values.
  */
  class DefaultBlockAssgnment
  {
    DefaultBlockAssgnment() = default;
    friend class BCRSMatrix;
  public:
    //! Assign a block value to the current entry. The value must be equal to the default constructed block.
    DefaultBlockAssgnment& operator=(const B& e);
    //! Assign a block to a field value. The value must be equal to the default constructed field type.
    DefaultBlockAssgnment& operator=(const typename FieldTraits<B>::field_type& f)
    requires (!std::same_as<B, typename FieldTraits<B>::field_type>);
  };

public:
  //! Type of the blocks stored in the matrix.
  using typename Matrix<B, SparseIndexRanges<I>, A>::block_type;
  //! Type of the indices used to access blocks.
  using typename Matrix<B, SparseIndexRanges<I>, A>::size_type;
  //! Type of the pattern.
  using typename Matrix<B, SparseIndexRanges<I>, A>::pattern_type;
  //! Type of the allocator used for memory management.
  using allocator_type = A;

  /**
   * \brief Available matrix construction modes.
   *
   * The values mirror the traditional BCRSMatrix build flow, but the
   * semantics are implemented on top of SparseIndexRanges.
   */
  enum BuildMode : uint_least8_t
  {
    /**
     * \brief Fill index sets sequentially, one row after another.
     *
     * It is a good fit when rows are naturally produced in order.
     *
     * Rows are initialized and then filled via @p createbegin and @p createend.
     * The iterator returned by @p createbegin represents the current row
     * and could be filled with column indices with the @p insert method.
     * The iterator is incrementable to the next row until all rows are
     * traversed, at which point comparing with the @p createend iterator
     * returns true and the pattern is finalized.
     *
     */
    row_wise = 0,
    /**
     * \brief Build entries in arbitrary order.
     *
     * In this mode, indices can be inserted in any order via the @p addindex
     * or @p entry methods. If rows sizes are known previous to
     * adding indices, it can be hited to the builders with the @p setrowsize
     * and @p incrementrowsize methods. Pattern creation is finalized
     * by calling either one of @p endindices or @p compress.
     */
    random = 1,
    /**
     * \brief Build entries in arbitrary order.
     *
     * Same as random build mode. This is kept for legacy reasons.
     */
    implicit = 1,
    /**
     * \brief No build mode selected, or matrix construction has finished.
     */
    unknown
  };

  struct CompressionStatistics
  {
    //! average number of non-zeroes per row.
    double avg;
    //! maximum number of non-zeroes per row.
    size_type maximum;
    //! total number of elements written to the overflow area during construction.
    size_type overflow_total;
    //! fraction of wasted memory resulting from non-used overflow area.
    double mem_ratio;
  };

  // inherit constructors TODO check that this works
  using Matrix<B, SparseIndexRanges<I>, A>::Matrix;

  using Matrix<B, SparseIndexRanges<I>, A>::operator=;

  using Matrix<B, SparseIndexRanges<I>, A>::get_allocator;

  //! Construct an BCRSMatrix from another Matrix.
  BCRSMatrix(const Matrix<B, SparseIndexRanges<I>, A>& other,
              const allocator_type& allocator = allocator_type())
    : Matrix<B, SparseIndexRanges<I>, A>(other, allocator)
  {
  }

  //! Move-construct an BCRSMatrix from another Matrix.
  BCRSMatrix(Matrix<B, SparseIndexRanges<I>, A>&& other,
              const allocator_type& allocator = allocator_type())
    : Matrix<B, SparseIndexRanges<I>, A>(std::move(other), allocator)
  {
  }

  /**
   * \brief Copy-construct an BCRSMatrix from another matrix.
   *
   * \warning Internal pattern builders are not copied, so the
   * resulting matrix is not in a build stage even if the source matrix is.
   *
   * \param other The matrix to copy from.
   * \param allocator The allocator to use for memory management.
   */
  BCRSMatrix(const BCRSMatrix& other,
              const allocator_type& allocator = allocator_type());
  //! Move-construct a matrix, optionally rebinding it to a different allocator.
  //! If allocators differ, block values may be moved into fresh storage.
  BCRSMatrix(BCRSMatrix&& other,
              const allocator_type& allocator = allocator_type());

  /**
   * \brief Copy-assign an BCRSMatrix from another matrix.
   *
   * \warning Internal pattern builders are not copied, so the
   * resulting matrix is not in a build stage even if the source matrix is.
   *
   * \param other The matrix to copy from.
   * \param allocator The allocator to use for memory management.
   */
  BCRSMatrix<B, I, A>& operator=(const BCRSMatrix& other);

  //! Move-assign matrix storage while respecting allocator propagation rules.
  BCRSMatrix<B, I, A>& operator=(BCRSMatrix&& other);

  /**
   * \brief Construct a matrix with a pre-selected build mode and dimensions.
   *
   *
   * \param rows Number of rows in the matrix.
   * \param cols Number of columns in the matrix.
   * \param bm Build mode of the pattern.
   */
  BCRSMatrix(size_type rows,
             size_type cols,
             BuildMode bm,
             const allocator_type& allocator = allocator_type());

  /**
   * \brief Construct a matrix with a pre-selected build mode, dimensions, and expected non-zero entries.
   *
   * \param rows Number of rows in the matrix.
   * \param cols Number of columns in the matrix.
   * \param nnz Expected number of non-zero entries in the matrix.
   * \param bm Build mode of the pattern.
   */
  BCRSMatrix(size_type rows,
             size_type cols,
             size_type nnz,
             BuildMode bm,
             const allocator_type& allocator = allocator_type());

  /**
   * \brief Construct a matrix with a pre-selected build mode, dimensions, and expected non-zero entries.
   *
   *
   * \param rows Number of rows in the matrix.
   * \param cols Number of columns in the matrix.
   * \param avg Expected average number of non-zero entries per row.
   * \param overflowRatio Expected ratio of overflow entries to total entries during construction.
   * \param bm Build mode of the pattern.
   */
  BCRSMatrix(size_type rows,
             size_type cols,
             size_type avg,
             double overflowRatio,
             BuildMode bm,
             const allocator_type& allocator = allocator_type());

  //! Swap the contents of two matrices, respecting allocator propagation rules.
  void swap(BCRSMatrix& other)
  {
    using std::swap;
    swap(static_cast<Matrix<B, SparseIndexRanges<I>, A>&>(*this),
         static_cast<Matrix<B, SparseIndexRanges<I>, A>&>(other));
    swap(builder_, other.builder_);
  }

  //! Swap the contents of two matrices, respecting allocator propagation rules.
  friend void swap(BCRSMatrix& lhs, BCRSMatrix& rhs)
  {
    lhs.swap(rhs);
  }

  /**
   * \brief Reset the matrix and select a new build mode.
   * \warning This clears the current matrix storage.
   * \param bm The new build mode.
   */
  void setBuildMode(BuildMode bm);

  //! Return the currently active build mode.
  BuildMode buildMode() const;

  //! Set the expected matrix dimensions for the currently selected build mode.
  void setSize(size_type rows, size_type cols, size_type nnz = 0);

  //! Random/implicit build API, mirroring the historic BCRSMatrix interface.

  /**
   * \brief Set the expected number of non-zero entries in a row for random/implicit build.
   *
   * \warning Use this before inserting indices to the pattern.
   *
   * \param row The zero-based index of the row to set the size for.
   * \param size The expected number of non-zero entries in the row.
   */
  void setrowsize(size_type row, size_type size);

  /**
   * \brief Increment the expected number of non-zero entries in a row for random/implicit build.
   *
   * \warning Use this before inserting indices to the pattern.
   *
   * \param row The zero-based index of the row to increment the size for.
   * \param increment The amount to increment the expected number of non-zero entries in the row by (default is 1).
   */
  void incrementrowsize(size_type row, size_type increment = 1);

  //! Finalize the row-size declaration phase.
  void endrowsizes() {}

  //! Insert the col index into a sparse row during implicit/random build.
  void addindex(size_type row, size_type col);
  //! Finalize the sparsity pattern and make the matrix ready for value access.
  void endindices();

  /**
   * \brief Set all column indices for row from the given iterator range on row-wise build mode.
   *
   * The iterator range has to be of the same length as the previously set row size.
   * The entries in the iterator range must be sorted and must not contain duplicate values.
   * This method will insert the indices in the given order.
   *
   * \warning Calling this method overwrites any previously set column indices!
   */
  template<typename It>
  void setIndicesNoSort(size_type row, It begin, It end) {
    DUNE_THROW(NotImplemented, "setIndicesNoSort is not implemented yet for BCRSMatrix row-wise build mode");
  }

  /**
   * \brief Set all column indices for row from the given iterator range row-wise build mode.
   *
   * The iterator range has to be of the same length as the previously set row size.
   * The entries in the iterator range do not have to be in any particular order, but
   * must not contain duplicate values.
   * This method will insert the indices and sort them afterwards.
   *
   * \warning Calling this method overwrites any previously set column indices!
   */
  template<typename It>
  void setIndices(size_type row, It begin, It end) {
    DUNE_THROW(NotImplemented, "setIndices is not implemented yet for BCRSMatrix row-wise build mode");
  }

  //! Row-wise build API.

  //! An iterator for row-wise construction of the matrix.
  class CreateIterator
    : public SequencedPatternBuilder::IndexRangeBuilder
  {
    using Base = typename SequencedPatternBuilder::IndexRangeBuilder;
    using EndSentinel = typename SequencedPatternBuilder::EndSentinel;

    //! Construct from the active row builder and owning matrix.
    CreateIterator(Base&& pattern_builder, BCRSMatrix* sm);

    CreateIterator(const CreateIterator&) = delete;
    CreateIterator& operator=(const CreateIterator&) = delete;
  public:
    //! Move-construct from another iterator.
    CreateIterator(CreateIterator&& other) noexcept;
    //! Move-assign from another iterator.
    CreateIterator& operator=(CreateIterator&& other) noexcept;
    //! Advance to the next row; when the last row is completed, finalize the matrix.
    CreateIterator& operator++();

  private:
    friend class BCRSMatrix;
    BCRSMatrix* matrix_;
  };

  //! Begin row-wise construction.
  CreateIterator createbegin();

  //! End sentinel for row-wise construction.
  typename SequencedPatternBuilder::EndSentinel createend();

  /**
   * \brief Insert an index in implicit/random build mode.
   *
   * \warning The object returned by this method allows assignment of block values
   * due to legacy reasons, but it does not actually store any values.
   * Only default constructed block values or field values are allowed to be assigned to it,
   * otherwise, the assignment throws a RangeError exception.
   *
   * \param row The zero-based index of the sparse row to insert the index into.
   * \param col The column index to insert into the row.
   * \return A proxy object that allows a (default constructed) assignment of a block value.
   */
  DefaultBlockAssgnment entry(size_type row, size_type col);

  //! Finalize implicit/random build and return compression statistics.
  CompressionStatistics compress();

private:
  template<class Builder>
  static Builder& getBuilder(BuilderVariant& builder,
                             std::string_view build_mode_name)
  {
    using BuilderPtrType = std::unique_ptr<Builder, AllocDeleter>;
    if (!std::holds_alternative<BuilderPtrType>(builder))
      DUNE_THROW(
        InvalidStateException,
        "Builder "
          << className<Builder>()
          << " can only be used when building a BCRSMatrix with BuildMode::"
          << build_mode_name);

    const auto& ptr = std::get<BuilderPtrType>(builder);
    if (!ptr)
      DUNE_THROW(
        InvalidStateException,
        "Builder "
          << className<Builder>()
          << " can only be called once the builder has been initialized");
    return *ptr;
  }

  SequencedPatternBuilder& getSequencedPatternBuilder()
  {
    return getBuilder<SequencedPatternBuilder>(builder_, "row_wise");
  }

  UnsequencedPatternBuilder& getUnsequencedPatternBuilder()
  {
    return getBuilder<UnsequencedPatternBuilder>(builder_, "implicit");
  }

  BuilderVariant builder_;
};

// Matrix implementation

template<class B, class I, class A>
BCRSMatrix<B, I, A>::BCRSMatrix(size_type n,
                                  size_type m,
                                  size_type nnz,
                                  BuildMode bm,
                                  const allocator_type& allocator)
  : Matrix<B, SparseIndexRanges<I>, A>(nullptr, allocator)
{
  setBuildMode(bm);
  setSize(n, m, nnz);
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::BCRSMatrix(size_type n,
                                  size_type m,
                                  BuildMode bm,
                                  const allocator_type& allocator)
  : BCRSMatrix(n, m, 0, bm, allocator)
{
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::BCRSMatrix(size_type n,
                                  size_type m,
                                  size_type avg,
                                  double overflowRatio,
                                  BuildMode bm,
                                  const allocator_type& allocator)
  : BCRSMatrix(n,
                m,
                static_cast<size_type>((avg * n) * overflowRatio),
                bm,
                allocator)
{
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::BCRSMatrix(const BCRSMatrix& other,
                                  const allocator_type& allocator)
  : Matrix<B, SparseIndexRanges<I>, A>(other, allocator)
{
  builder_ = std::monostate{};
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::BCRSMatrix(BCRSMatrix&& other,
                                  const allocator_type& allocator)
  : Matrix<B, SparseIndexRanges<I>, A>(std::move(other), allocator)
{
  builder_ = std::move(other.builder_);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::setBuildMode(BuildMode bm)
{
  *this = BCRSMatrix(get_allocator());
  switch (bm) {
    case BuildMode::row_wise:
      builder_ =
        std::unique_ptr<SequencedPatternBuilder, AllocDeleter>(
          nullptr, AllocDeleter{ get_allocator() });
      break;
    case BuildMode::implicit: // same as BuildMode::random
      builder_ =
        std::unique_ptr<UnsequencedPatternBuilder, AllocDeleter>(
          nullptr, AllocDeleter{ get_allocator() });
      break;
    case BuildMode::unknown:
      builder_ = std::monostate{};
      break;
  }
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::buildMode() const -> BuildMode
{
  return std::visit(
    overload(
      [](std::monostate) { return BuildMode::unknown; },
      [](const std::unique_ptr<SequencedPatternBuilder,
                               AllocDeleter>&) { return BuildMode::row_wise; },
      [](const std::unique_ptr<UnsequencedPatternBuilder,
                               AllocDeleter>&) { return BuildMode::implicit; }),
    builder_);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::setSize(size_type rows, size_type cols, size_type nnz)
{
  using seq_builder_alloc_type = typename std::allocator_traits<
    allocator_type>::template rebind_alloc<SequencedPatternBuilder>;
  using unseq_builder_alloc_type = typename std::allocator_traits<
    allocator_type>::template rebind_alloc<UnsequencedPatternBuilder>;
  const auto void_alloc = void_alloc_type(get_allocator());
  size_type avg = (rows > 0) ? (nnz / rows) : 0;
  std::visit(
    overload(
      [](std::monostate) {
        DUNE_THROW(InvalidStateException,
                   "Cannot set size of BCRSMatrix with unknown build mode");
      },
      [balloc = seq_builder_alloc_type(get_allocator()),
       void_alloc,
       rows,
       cols,
       avg](std::unique_ptr<SequencedPatternBuilder, AllocDeleter>&
              builder) mutable {
        builder.reset(
          std::allocator_traits<seq_builder_alloc_type>::allocate(balloc, 1));
        std::allocator_traits<seq_builder_alloc_type>::construct(
          balloc, builder.get(), rows, cols, avg, void_alloc);
      },
      [balloc = unseq_builder_alloc_type(get_allocator()),
       void_alloc,
       rows,
       cols,
       avg](std::unique_ptr<UnsequencedPatternBuilder, AllocDeleter>&
              builder) mutable {
        builder.reset(
          std::allocator_traits<unseq_builder_alloc_type>::allocate(balloc, 1));
        std::allocator_traits<unseq_builder_alloc_type>::construct(
          balloc, builder.get(), rows, cols, avg, void_alloc);
      }),
    builder_);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::setrowsize(size_type row, size_type size)
{
  getUnsequencedPatternBuilder().setSize(row, size);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::incrementrowsize(size_type row, size_type increment)
{
  getUnsequencedPatternBuilder().incrementSize(row, increment);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::addindex(size_type row, size_type col)
{
  getUnsequencedPatternBuilder().addIndex(row, col);
}

template<class B, class I, class A>
void
BCRSMatrix<B, I, A>::endindices()
{
  using pattern_alloc_type = typename std::allocator_traits<
    allocator_type>::template rebind_alloc<SparseIndexRanges<I>>;
  pattern_alloc_type pattern_alloc(get_allocator());
  std::shared_ptr<pattern_type> pt = std::allocate_shared<pattern_type>(
    pattern_alloc, std::move(getUnsequencedPatternBuilder()));
  *this = Matrix<B, SparseIndexRanges<I>, A>(std::move(pt), get_allocator());
  builder_ = std::monostate{};
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::CreateIterator::CreateIterator(Base&& pattern_builder,
                                                     BCRSMatrix* sm)
  : Base(std::move(pattern_builder))
  , matrix_(sm)
{
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>::CreateIterator::CreateIterator(
  CreateIterator&& other) noexcept
  : Base(std::move(other))
  , matrix_(std::exchange(other.matrix_, nullptr))
{
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::CreateIterator::operator=(CreateIterator&& other) noexcept
  -> CreateIterator&
{
  if (this != &other) {
    Base::operator=(std::move(other));
    matrix_ = std::exchange(other.matrix_, nullptr);
  }
  return *this;
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::CreateIterator::operator++() -> CreateIterator&
{
  Base::operator++();
  assert(matrix_);
  if (*this == EndSentinel()) {
    // build pattern and allocate data when we have reached the end of the index
    // sets
    using pattern_alloc_type =
      typename block_alloc_traits::template rebind_alloc<pattern_type>;
    pattern_alloc_type pattern_alloc(matrix_->get_allocator());
    std::shared_ptr<pattern_type> pt = std::allocate_shared<pattern_type>(
      pattern_alloc, std::move(matrix_->getSequencedPatternBuilder()));
    *matrix_ =
      Matrix<B, SparseIndexRanges<I>, A>(std::move(pt), matrix_->get_allocator());
    matrix_->builder_ = std::monostate{};
  }
  return *this;
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::createbegin() -> CreateIterator
{
  return { getSequencedPatternBuilder().begin(), this };
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::createend() ->
  typename SequencedPatternBuilder::EndSentinel
{
  return {};
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::entry(size_type row, size_type col)
  -> DefaultBlockAssgnment
{
  getUnsequencedPatternBuilder().addIndex(row, col);
  return {};
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::compress() -> CompressionStatistics
{
  endindices();

  size_type max_pattern_size = 0;
  for (const auto& index_range : this->pattern())
    max_pattern_size =
      std::max<size_type>(max_pattern_size, index_range.size());
  return CompressionStatistics{ /*avg = */ this->pattern().count() /
                                  static_cast<double>(this->pattern().size()),
                                /* maximum = */ max_pattern_size,
                                /* overflow_total = */ 0,
                                /* mem_ratio = */ 1.0 };
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::DefaultBlockAssgnment::operator=(const B& e)
  -> DefaultBlockAssgnment&
{
  if (e != B{})
    DUNE_THROW(RangeError,
               "BCRSMatrix with BuildMode::row_wise only supports default "
               "constructed entries, but got "
                 << e);
  return *this;
}

template<class B, class I, class A>
auto
BCRSMatrix<B, I, A>::DefaultBlockAssgnment::operator=(const typename FieldTraits<B>::field_type& e)
-> DefaultBlockAssgnment&
requires (!std::same_as<B, typename FieldTraits<B>::field_type>)
{
  if (e != typename FieldTraits<B>::field_type{})
    DUNE_THROW(RangeError,
               "BCRSMatrix with BuildMode::row_wise only supports default "
               "constructed entries, but got "
                 << e);
  return *this;
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>&
BCRSMatrix<B, I, A>::operator=(const BCRSMatrix& other)
{
  if (this != &other) {
    static_cast<Matrix<B, SparseIndexRanges<I>, A>&>(*this) =
      static_cast<const Matrix<B, SparseIndexRanges<I>, A>&>(other);
    builder_ = std::monostate{};
  }
  return *this;
}

template<class B, class I, class A>
BCRSMatrix<B, I, A>&
BCRSMatrix<B, I, A>::operator=(BCRSMatrix&& other)
{
  if (this != &other) {
    static_cast<Matrix<B, SparseIndexRanges<I>, A>&>(*this) = std::move(other);
    builder_ = std::move(other.builder_);
  }
  return *this;
}

} //namespace Dune::Experimental

namespace Dune {
template<class B, class I, class A>
struct FieldTraits<Experimental::BCRSMatrix<B, I, A>>
  : public FieldTraits<typename Experimental::BCRSMatrix<B, I, A>::field_type>
{};

} // namespace Dune

#endif // DUNE_ISTL_EXPERIMENTAL_BCRSMATRIX_HH
