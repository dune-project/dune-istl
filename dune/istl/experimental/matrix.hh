// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_EXPERIMENTAL_MATRIX_HH
#define DUNE_ISTL_EXPERIMENTAL_MATRIX_HH

#include "sparseindexranges.hh"
#include "matrixview.hh"

#include <dune/common/std/type_traits.hh>
#include <dune/common/std/no_unique_address.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/typetraits.hh>

#include <cassert>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Dune::Experimental
{
  /**
   * \brief Owning matrix container built on top of MatrixView.
   *
   * Matrix stores block values and owns a pattern while exposing
   * the matrix interface from MatrixView.
   *
   * In contrast to MatrixView, this class owns both resources:
   * - block storage for all structurally present entries,
   * - a shared pointer to the immutable sparsity pattern.
   *
   * \tparam B Block value type.
   * \tparam P Pattern type (See MatrixView).
   * \tparam A Allocator type used for block and pattern storage.
   */
  template <class B, class P, class A = std::allocator<B>>
  class Matrix : public MatrixView<B*, P>
  {
    using block_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<B>;
    using block_alloc_traits = std::allocator_traits<block_alloc_type>;

  public:
    //! Block type of matrix entries.
    using typename MatrixView<B*, P>::block_type;
    //! Size/index type provided by the pattern type.
    using typename MatrixView<B*, P>::size_type;
    //! Pattern type.
    using typename MatrixView<B*, P>::pattern_type;
    //! Allocator type used by this container.
    using allocator_type = A;

    /** \brief Construct an empty matrix with allocator \p allocator.
     *
     * The resulting matrix has no pattern and no allocated block storage.
     */
    Matrix(const allocator_type &allocator = allocator_type());

    /** \brief Construct from a shared pattern.
     *
     * If \p pattern is non-null, block storage is allocated and value-initialized
     * for \c pattern->count() entries.
     */
    Matrix(std::shared_ptr<const pattern_type> pattern, const allocator_type &allocator = allocator_type());

    /** \brief Construct from an rvalue pattern.
     *
     * The pattern is moved into shared owned storage and corresponding block
     * storage is allocated and value-initialized.
     */
    Matrix(pattern_type &&pattern, const allocator_type &allocator = allocator_type());

    /** \brief Copy constructor.
     *
     * Shares the sparsity pattern and deep-copies block values.
     */
    Matrix(const Matrix &other);

    /** \brief Copy constructor.
     *
     * Shares the pattern and deep-copies block values.
     */
    Matrix(const Matrix &other, const allocator_type &allocator);

    /** \brief Copy assignment.
     *
     * Provides strong exception safety for block storage replacement.
     */
    Matrix &operator=(const Matrix &other);

    /** \brief Move constructor.
     *
     * Transfers owned resources from \p other.
     */
    Matrix(Matrix &&other);

    /** \brief Move assignment.
     *
     * Transfers owned resources from \p other according to allocator propagation
     * rules.
     */
    Matrix &operator=(Matrix &&other);

    /** \brief Move construct with explicit allocator.
     *
     * If allocators compare equal, resources are stolen. Otherwise, pattern is
     * shared and block values are moved into newly allocated storage.
     */
    Matrix(Matrix &&other, const allocator_type &allocator);

    using MatrixView<B*, P>::operator=;

    //! Destructor releases owned block storage.
    ~Matrix();

    /** \brief Return the allocator currently associated with this matrix. */
    allocator_type get_allocator() const;

    //! Swap contents of this matrix with \p other, honoring allocator swap propagation traits.
    void swap(Matrix &other);

    //! Swap contents of \p lhs and \p rhs, honoring allocator swap propagation traits.
    friend void swap(Matrix &lhs, Matrix &rhs)
    {
      lhs.swap(rhs);
    }

    //! Get shared storage of the pattern.
    std::shared_ptr<const pattern_type> patternStorage() const { return pattern_; }

  private:

    // matrix view getters/setters
    using MatrixView<B*, P>::blockIter;
    using MatrixView<B*, P>::patternPtr;
    using MatrixView<B*, P>::resetView;

    // matrix view types
    using typename MatrixView<B*, P>::block_iter_type;
    using typename MatrixView<B*, P>::block_const_iter_type;

    // Release owned block storage if it exists; pattern ownership is handled separately.
    void resetData();

    //! Construct a block at \p ptr using the allocator \p block_alloc and the allocator \p alloc.
    template<class BlockAlloc, class Alloc, class... Args>
    static void constructBlock(BlockAlloc& block_alloc, block_iter_type ptr, const Alloc& alloc, Args&&... args);

    //! Allocate block storage for \p count blocks using the allocator \p alloc and the construction function \p construct_fn.
    template<class Alloc, class ConstructFn>
    block_iter_type allocateData(const Alloc& alloc, size_type count, ConstructFn&& construct_fn);

    //! Allocate block storage for \p count blocks using the allocator \p alloc and the construction function \p construct_fn.
    //! If construction fails, the \p restore_fn is called for each constructed block to restore the source before deallocation.
    template<class Alloc, class ConstructFn, class RestoreFn>
    block_iter_type allocateData(const Alloc& alloc, size_type count, ConstructFn&& construct_fn, RestoreFn&& restore_fn);

    //! Allocate block storage for the current pattern and value-initialize all blocks.
    void allocateDataDefault();

    //! Allocate block storage for the current pattern and copy-construct all blocks from \p other_data.
    void allocateDataCopy(block_const_iter_type other_data);

    //! Allocate block storage for the current pattern and move-construct all blocks from \p other_data.
    void allocateDataMove(block_iter_type other_data);

    //! Deallocate block storage if it exists.
    void deallocateData();

    // storage of index sets, shared ownership with this class
    std::shared_ptr<const pattern_type> pattern_;

    // The allocator used for memory management
    DUNE_NO_UNIQUE_ADDRESS allocator_type alloc_;
  };

  namespace Impl
  {
    // Lightweight replacement of std::uses_allocator_construction_args without overloads to std::pair
    // This is needed to support libc++ versions prior to 16.
    template<class T, class Alloc, class... Args>
    constexpr auto usesAllocatorConstructionArgs(const Alloc& alloc, Args&&... args)
    {
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION <= 16000
      if constexpr (!std::uses_allocator_v<std::remove_cv_t<T>, Alloc> && std::is_constructible_v<T, Args...>) {
        return std::forward_as_tuple(std::forward<Args>(args)...);
      } else if constexpr (std::uses_allocator_v<std::remove_cv_t<T>, Alloc>
                           && std::is_constructible_v<T, std::allocator_arg_t, const Alloc&, Args...>) {
        return std::tuple<std::allocator_arg_t, const Alloc&, Args&&...>(
          std::allocator_arg, alloc, std::forward<Args>(args)...);
      } else if constexpr (std::uses_allocator_v<std::remove_cv_t<T>, Alloc>
                           && std::is_constructible_v<T, Args..., const Alloc&>) {
        return std::forward_as_tuple(std::forward<Args>(args)..., alloc);
      } else {
        static_assert(Dune::AlwaysFalse<T>::value,
                      "Type is not allocator-constructible for the provided allocator and arguments");
      }
# else
      return std::uses_allocator_construction_args<T>(alloc, std::forward<Args>(args)...);
#endif
    }
  } // namespace Impl

  // Matrix implementation

  template <class B, class P, class A>
  void Matrix<B, P, A>::resetData()
  {
    if (blockIter())
      deallocateData();
  }

  template <class B, class P, class A>
  template<class BlockAlloc, class Alloc, class... Args>
  void Matrix<B, P, A>::constructBlock(BlockAlloc& block_alloc, block_iter_type ptr, const Alloc& alloc, Args&&... args)
  {
    std::apply([&]<class... Xs>(Xs&&... xs) {
      block_alloc_traits::construct(block_alloc, ptr, std::forward<Xs>(xs)...);
    }, Impl::usesAllocatorConstructionArgs<block_type>(alloc, std::forward<Args>(args)...));
  }

  template <class B, class P, class A>
  template<class Alloc, class ConstructFn>
  auto Matrix<B, P, A>::allocateData(const Alloc& alloc, size_type count, ConstructFn&& construct_fn) -> block_iter_type
  {
    return allocateData(alloc, count, std::forward<ConstructFn>(construct_fn), [](size_type, block_type&) {});
  }

  template <class B, class P, class A>
  template<class Alloc, class ConstructFn, class RestoreFn>
  auto Matrix<B, P, A>::allocateData(const Alloc& alloc, size_type count, ConstructFn&& construct_fn, RestoreFn&& restore_fn) -> block_iter_type
  {
    // Strong exception guarantee: either fully constructed storage is returned
    // or all partially constructed elements are rolled back and deallocated.
    block_alloc_type block_alloc(alloc);
    block_iter_type data = block_alloc_traits::allocate(block_alloc, count);
    size_type constructed = 0;
    try {
      for (; constructed < count; ++constructed)
        construct_fn(block_alloc, data + constructed, constructed);
    } catch (...) {
      if constexpr (not std::is_trivially_destructible_v<block_type>)
        for (size_type i = 0; i < constructed; ++i) {
          restore_fn(i, data[i]);
          block_alloc_traits::destroy(block_alloc, std::addressof(data[i]));
        }
      block_alloc_traits::deallocate(block_alloc, data, count);
      throw;
    }
    return data;
  }

  template <class B, class P, class A>
  void Matrix<B, P, A>::allocateDataDefault()
  {
    assert(not blockIter());
    assert(patternPtr());
    assert(patternPtr() == pattern_.get());
    const size_type count = pattern_->count();
    block_iter_type data_ptr = allocateData(alloc_, count, [this](auto& block_alloc, block_iter_type ptr, size_type) {
      constructBlock(block_alloc, ptr, alloc_);
    });
    resetView(data_ptr, pattern_.get());
  }

  template <class B, class P, class A>
  void Matrix<B, P, A>::allocateDataCopy(block_const_iter_type other_data)
  {
    assert(not blockIter());
    assert(patternPtr());
    assert(patternPtr() == pattern_.get());
    const size_type count = pattern_->count();
    block_iter_type data_ptr = allocateData(alloc_, count, [this, other_data](auto& block_alloc, block_iter_type ptr, size_type i) {
      constructBlock(block_alloc, ptr, alloc_, other_data[i]);
    });
    resetView(data_ptr, pattern_.get());
  }

  template <class B, class P, class A>
  void Matrix<B, P, A>::allocateDataMove(block_iter_type other_data)
  {
    assert(not blockIter());
    assert(patternPtr());
    assert(patternPtr() == pattern_.get());
    const size_type count = pattern_->count();
    block_iter_type data_ptr = allocateData(
      alloc_,
      count,
      [this, other_data](auto& block_alloc, block_iter_type ptr, size_type i) {
        constructBlock(block_alloc, ptr, alloc_, std::move(other_data[i]));
      },
      [other_data](size_type i, block_type& block) {
        other_data[i] = std::move(block);
      });
    resetView(data_ptr, pattern_.get());
  }

  template <class B, class P, class A>
  void Matrix<B, P, A>::deallocateData()
  {
    assert(blockIter());
    assert(patternPtr());
    assert(patternPtr() == pattern_.get());
    block_alloc_type block_alloc(alloc_);
    if constexpr (not std::is_trivially_destructible_v<block_type>)
      for (auto& block : std::span(blockIter(), pattern_->count()))
        block_alloc_traits::destroy(block_alloc, std::addressof(block));
    block_alloc_traits::deallocate(block_alloc, blockIter(), pattern_->count());
    resetView(nullptr, pattern_.get());
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(const allocator_type &allocator)
    : Matrix(nullptr, allocator)
  {
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(std::shared_ptr<pattern_type const> pattern, const allocator_type &allocator)
      : MatrixView<B*, P>::MatrixView{nullptr, pattern.get()}
      , pattern_(std::move(pattern))
      , alloc_(allocator)
  {
    if (patternPtr())
      allocateDataDefault();
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(pattern_type &&pattern, const allocator_type &allocator)
      : Matrix(nullptr, allocator)
  {
    using pattern_alloc_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<P>;
    pattern_alloc_type pattern_alloc(alloc_);
    pattern_ = std::allocate_shared<const pattern_type>(pattern_alloc, std::move(pattern));
    resetView(nullptr, pattern_.get());
    if (patternPtr())
      allocateDataDefault();
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::~Matrix()
  {
    resetData();
    resetView(nullptr, nullptr);
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(const Matrix &other)
    : Matrix(other, std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.alloc_))
  {}

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(const Matrix &other, const allocator_type &allocator)
    : Matrix(other.pattern_, allocator)
  {
    resetData();
    if (patternPtr())
      allocateDataCopy(other.blockIter());
  }

  template <class B, class P, class A>
  auto Matrix<B, P, A>::get_allocator() const -> allocator_type
  {
    return alloc_;
  }

  template <class B, class P, class A>
  void Matrix<B, P, A>::swap(Matrix &other)
  {
    if constexpr (!std::allocator_traits<allocator_type>::propagate_on_container_swap::value
                  && !std::allocator_traits<allocator_type>::is_always_equal::value)
      if (alloc_ != other.alloc_)
        DUNE_THROW(InvalidStateException, "Cannot swap Matrix instances with unequal allocators when propagate_on_container_swap is false");

    using std::swap;
    swap(pattern_, other.pattern_);
    pattern_type const * tmp_pattern_ptr = patternPtr();
    block_iter_type tmp_data_iter = blockIter();
    resetView(other.blockIter(), other.patternPtr());
    other.resetView(tmp_data_iter, tmp_pattern_ptr);
    if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value)
      swap(alloc_, other.alloc_);
  }

  template <class B, class P, class A>
  Matrix<B, P, A> &Matrix<B, P, A>::operator=(const Matrix &other)
  {
    if (this != &other) {
      allocator_type target_alloc = alloc_;
      if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value)
        target_alloc = other.alloc_;

      auto new_pattern = other.pattern_;
      auto* new_pattern_view = new_pattern.get();
      block_iter_type new_block_ptr = nullptr;

      if (new_pattern_view)
        new_block_ptr = allocateData(target_alloc, new_pattern_view->count(), [&](auto& block_alloc, block_iter_type ptr, size_type i) {
          constructBlock(block_alloc, ptr, target_alloc, other.blockIter()[i]);
        });

      resetData();

      if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value)
        alloc_ = other.alloc_;

      pattern_ = std::move(new_pattern);
      resetView(new_block_ptr, new_pattern_view);
    }
    return *this;
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(Matrix &&other)
    : MatrixView<B*, P>(nullptr, nullptr)
  {
    alloc_ = std::move(other.alloc_);
    pattern_ = std::move(other.pattern_);
    resetView(other.blockIter(), other.patternPtr());
    other.resetView(nullptr, nullptr);
  }

  template <class B, class P, class A>
  Matrix<B, P, A>::Matrix(Matrix &&other, const allocator_type &allocator)
    : MatrixView<B*, P>(nullptr, nullptr)
    , alloc_(allocator)
  {
    // If allocators compare equal we can steal storage directly, otherwise we
    // preserve source validity by moving elements into newly allocated storage.
    if (alloc_ == other.alloc_) {
      pattern_ = std::move(other.pattern_);
      resetView(other.blockIter(), other.patternPtr());
      other.resetView(nullptr, nullptr);
    } else {
      pattern_ = other.pattern_;
      resetView(nullptr, pattern_.get());
      if (other.blockIter())
        allocateDataMove(other.blockIter());
    }
  }

  template <class B, class P, class A>
  Matrix<B, P, A> &Matrix<B, P, A>::operator=(Matrix &&other)
  {
    if (this != &other)
    {
      // Fast path: allocator propagation allows full ownership transfer.
      if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
        resetData();
        alloc_ = std::move(other.alloc_);
        pattern_ = std::move(other.pattern_);
        resetView(other.blockIter(), other.patternPtr());
        other.resetView(nullptr, nullptr);
      // Also fast when allocators match even without propagation.
      } else if (alloc_ == other.alloc_) {
        resetData();
        pattern_ = std::move(other.pattern_);
        resetView(other.blockIter(), other.patternPtr());
        other.resetView(nullptr, nullptr);
      } else {
        // Conservative path: rebuild data with this allocator, moving entries
        // one by one and restoring the source on construction failure.
        auto new_pattern = other.pattern_;
        auto* new_pattern_view = new_pattern.get();
        block_iter_type new_block_ptr = nullptr;

        if (new_pattern_view && other.blockIter())
          new_block_ptr = allocateData(
            alloc_,
            new_pattern_view->count(),
            [&](auto& block_alloc, block_iter_type ptr, size_type i) {
              constructBlock(block_alloc, ptr, alloc_, std::move(other.blockIter()[i]));
            },
            [&](size_type i, block_type& block) {
              other.blockIter()[i] = std::move(block);
            });

        resetData();

        pattern_ = std::move(new_pattern);
        resetView(new_block_ptr, new_pattern_view);
      }
    }
    return *this;
  }
} // namespace Dune::Experimental

namespace Dune {

template <class B, class P, class A>
struct FieldTraits<Experimental::Matrix<B, P, A>>
    : public FieldTraits<typename Experimental::Matrix<B, P, A>::field_type>
{
};

} // namespace Dune

#endif // DUNE_ISTL_EXPERIMENTAL_MATRIX_HH
