// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_EXPERIMENTAL_MATRIXVIEW_HH
#define DUNE_ISTL_EXPERIMENTAL_MATRIXVIEW_HH

#include <dune-istl-config.hh> // DUNE_ISTL_WITH_CHECKING_NOEXCEPT

#include "vectorview.hh"

#include <dune/common/std/type_traits.hh>
#include <dune/common/std/no_unique_address.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/scalarvectorview.hh>

#include <cfenv>
#include <limits>
#include <memory>
#include <ranges>
#include <span>

namespace Dune::Experimental
{

  /**
   * \brief Non-owning matrix view over external block storage and pattern.
   *
   * MatrixView binds two external objects:
   * - a random-access iterator to block values,
   * - a pattern that defines row offsets and column indices.
   *
   * The block iterator is expected to point to at least \c pattern().count()
   * blocks, which are interpreted in row-major order consistent with
   * index ranges of \c pattern_type. That is, entries of row \f$i\f$ start at
   * an offset given by \c pattern().offset(i) and follow the column indices
   * in \c pattern()[i].
   *
   * The view does not own memory. The caller must ensure that both the data
   * range and the pattern outlive the view. The pattern is treated as
   * read-only for the lifetime of the view.
   *
   * A pattern is a sized range of BorrowedOrderedIndexRange with three extra methods:
   * - \c offset(i) counts the total number of entries in rows \f$0\f$ to \f$i-1\f$.
   * - \c count() counts the total number of entries in the matrix.
   * - \c range() returns the logical range of row indices.
   *
   * \tparam B Random-access iterator type for matrix blocks.
   * \tparam P Pattern type with row ranges and offsets.
   */
  template <std::random_access_iterator B, class P>
  class MatrixView
  {
  public:
    //! The iterator type used to access the blocks of the matrix.
    using block_iter_type = B;
    //! The const iterator type used to access the blocks of the matrix.
#if __cpp_lib_ranges_as_const <= 202311L
    class block_const_iter_type;
#else
    using block_const_iter_type = std::basic_const_iterator<B>;
#endif

    //! The type used for the sparsity pattern.
    using pattern_type = P;
    //! The type used for the index set of rows.
    using row_index_range_type = std::ranges::range_value_t<pattern_type>;
    //! The type of blocks in the matrix
    using block_type = std::iter_value_t<block_iter_type>;
    //! The type of blocks in the matrix
    using value_type = block_type;
    //! The type used to count elements in the row and column index sets
    using size_type = std::ranges::range_size_t<pattern_type>;
    //! The type of the field
    using field_type = typename FieldTraits<block_type>::field_type;
    //! The type representing a mutable matrix row
    using row_type = VectorView<block_iter_type, row_index_range_type>;
    //! The type representing a const matrix row
    using const_row_type = VectorView<block_const_iter_type, row_index_range_type>;
    //! The type of mutable matrix iterators over the rows
    class Iterator;
    //! The type of const matrix iterators over the rows
    class ConstIterator;
    //! The type of mutable matrix iterators over the rows
    using iterator = Iterator;
    //! The type of const matrix iterators over the rows
    using const_iterator = ConstIterator;
    //! The type of mutable matrix iterators over the rows
    using RowIterator = Iterator;
    //! The type of const matrix iterators over the rows
    using ConstRowIterator = ConstIterator;
    //! The type of mutable row iterators over the column blocks
    using ColIterator = typename row_type::iterator;
    //! The type of const row iterators over the column blocks
    using ConstColIterator = typename const_row_type::const_iterator;

    /**
     * \brief Construct a view from data iterator and sparsity pattern.
     *
     * \param data Iterator to the first stored block.
     * \param pattern_ptr Pointer to the external sparsity pattern.
     */
    MatrixView(block_iter_type block_iter, pattern_type const* pattern_ptr) noexcept
      : block_iter_(block_iter), pattern_ptr_(pattern_ptr)
    {}

    /** \brief Return a mutable view of row \p i. */
    row_type operator[](size_type i) { return begin()[i]; }

    /** \brief Return a const view of row \p i. */
    const_row_type operator[](size_type i) const { return begin()[i]; }

    /** \brief Return iterator to the first row. */
    Iterator begin()  { return {block_iter_, pattern_ptr_, 0}; }

    /** \brief Return iterator one past the last row. */
    Iterator end() { return {block_iter_, pattern_ptr_, pattern_ptr_->size()}; }

    /** \brief Return const iterator to the first row. */
    ConstIterator begin() const { return cbegin(); }

    /** \brief Return const iterator one past the last row. */
    ConstIterator end() const { return cend(); }

    /** \brief Return const iterator to the first row. */
    ConstIterator cbegin() const { return {{block_iter_, pattern_ptr_, 0}}; }

    /** \brief Return const iterator one past the last row. */
    ConstIterator cend() const { return {{block_iter_, pattern_ptr_, pattern_ptr_->size()}}; }

    /** \brief Compute \f$this \leftarrow this + \alpha b\f$. */
    template <class OtherB, class OtherI>
    MatrixView &axpy(field_type alpha, const MatrixView<OtherB, OtherI> &b);

    /** \brief Compute \f$y = Ax\f$. */
    template <class X, class Y>
    void mv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= Ax\f$. */
    template <class X, class Y>
    void umv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{-}= Ax\f$. */
    template <class X, class Y>
    void mmv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= \alpha Ax\f$. */
    template <class X, class Y>
    void usmv(const field_type &alpha, const X &x, Y &y) const;

    /** \brief Compute \f$y = A^T x\f$. */
    template <class X, class Y>
    void mtv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= A^T x\f$. */
    template <class X, class Y>
    void umtv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{-}= A^T x\f$. */
    template <class X, class Y>
    void mmtv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= \alpha A^T x\f$. */
    template <class X, class Y>
    void usmtv(const field_type &alpha, const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= A^H x\f$ (Hermitian transpose). */
    template <class X, class Y>
    void umhv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{-}= A^H x\f$ (Hermitian transpose). */
    template <class X, class Y>
    void mmhv(const X &x, Y &y) const;

    /** \brief Compute \f$y \mathrel{+}= \alpha A^H x\f$ (Hermitian transpose). */
    template <class X, class Y>
    void usmhv(const field_type &alpha, const X &x, Y &y) const;

    /** \brief Return \f$\|A\|_F^2\f$ (sum of squared block Frobenius norms). */
    typename FieldTraits<field_type>::real_type frobenius_norm2() const;

    /** \brief Return the Frobenius norm \f$\|A\|_F\f$. */
    typename FieldTraits<field_type>::real_type frobenius_norm() const;

    /** \brief Return the block-aware infinity norm. */
    typename FieldTraits<field_type>::real_type infinity_norm() const;

    /** \brief Return simplified infinity norm (Manhattan norm for complex values). */
    typename FieldTraits<field_type>::real_type infinity_norm_real() const;

    /**
     * \brief Test whether position \f$(i,j)\f$ is present in the sparsity pattern.
     *
     * This checks structural existence only. It does not inspect value magnitude.
     * \throw BCRSMatrixError if \f$i\f$ or \f$j\f$ is out of range if DUNE_ISTL_WITH_CHECKING is defined.
     */
    [[nodiscard]] bool exists (size_type i, size_type j) const DUNE_ISTL_WITH_CHECKING_NOEXCEPT
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=N()) DUNE_THROW(BCRSMatrixError,"row index out of range");
      if (j<0 || j>=M()) DUNE_THROW(BCRSMatrixError,"column index out of range");
#endif
      return this->operator[](i).contains(j);
    }

    /** \brief Number of block rows. */
    size_type N() const { return pattern_ptr_ ? pattern().size() : 0; }

    /** \brief Number of block columns. */
    size_type M() const { return pattern_ptr_ ? pattern().range().size() : 0; }

    /**
     * \brief Number of stored block entries.
     *
     * This is the number of structurally present entries, i.e., entries that may
     * be nonzero.
     */
    size_type nonzeroes() const { return pattern_ptr_ ? pattern().count() : 0; }

    //===== vector space arithmetic

    /** \brief Scale all stored blocks by \p k. */
    MatrixView &operator*=(const field_type &k);

    /** \brief Divide all stored blocks by \p k. */
    MatrixView &operator/=(const field_type &k);

    /** \brief Add another matrix view to this one.
     *
     * \param b The matrix to add to this one. Its sparsity pattern
     * has to be subset of the sparsity pattern of this matrix.
     */
    template <class OtherB, class OtherI>
    MatrixView &operator+=(const MatrixView<OtherB, OtherI> &b);

    /** \brief Subtract another matrix view from this one.
     *
     * \param b The matrix to subtract from this one. Its sparsity pattern
     * has to be subset of the sparsity pattern of this matrix.
     */
    template <class OtherB, class OtherI>
    MatrixView &operator-=(const MatrixView<OtherB, OtherI> &b);

    /** \brief Assign all stored blocks to scalar \p k. */
    MatrixView &operator=(const field_type &k);

    const pattern_type& pattern() const {
      assert(pattern_ptr_);
      return *pattern_ptr_;
    }

  private:
    template <class X, class Y, class F, bool ClearRow = false>
    void umvImpl(const X &x, Y &y, F &&op, std::bool_constant<ClearRow> = {}) const;

    template <class X, class Y, class F>
    void mtvImpl(const X &x, Y &y, F &&op) const;

  protected:

    //! Reset the view to point to \p block_iter and \p pattern_ptr.
    void resetView(block_iter_type block_iter, pattern_type const* pattern_ptr)
    {
      block_iter_ = block_iter;
      pattern_ptr_ = pattern_ptr;
    }

    //! Return the underlying block iterator.
    block_iter_type blockIter() {
      return block_iter_;
    }

    //! Return the underlying block iterator as a const iterator.
    block_const_iter_type blockIter() const {
      return block_const_iter_type{block_iter_};
    }

    //! Return the underlying pattern pointer.
    pattern_type const* patternPtr() const {
      return pattern_ptr_;
    }

  private:
    // View to underlying data
    block_iter_type block_iter_ = {};
    // View to underlying sparsity pattern.
    pattern_type const* pattern_ptr_ = nullptr;
  };
} // namespace Dune::Experimental


namespace Dune {

  template <std::random_access_iterator B, class I>
  struct FieldTraits<Experimental::MatrixView<B, I>>
      : public FieldTraits<typename Experimental::MatrixView<B, I>::block_type>
  {
  };

} // namespace Dune

namespace Dune::Experimental {

#if __cpp_lib_ranges_as_const <= 202311L

  namespace Impl {
    template<std::random_access_iterator B>
    using iter_const_reference_t = std::common_reference_t<const std::iter_value_t<B>&&, std::iter_reference_t<B>>;
  }

  /**
   * \brief Const block iterator adapter.
   *
   * This adapter wraps the mutable block iterator type \c B and exposes
   * const-qualified element access for const matrix row views.
   */
  template <std::random_access_iterator B, class I>
  class MatrixView<B, I>::block_const_iter_type
    : public Dune::IteratorFacade<block_const_iter_type,
                                  std::random_access_iterator_tag,
                                  std::remove_reference_t<Impl::iter_const_reference_t<B>>,
                                  Impl::iter_const_reference_t<B>>
  {
    using Facade =
      Dune::IteratorFacade<block_const_iter_type,
                           std::random_access_iterator_tag,
                           std::remove_reference_t<Impl::iter_const_reference_t<B>>,
                           Impl::iter_const_reference_t<B>>;

  public:
    //! The type used for references to the components
    using reference = typename Facade::reference;
    //! The type used for differences between iterators
    using difference_type = typename Facade::difference_type;

    //! Default constructor.
    constexpr block_const_iter_type() noexcept = default;

    //! Construct a const iterator from a mutable iterator.
    constexpr block_const_iter_type(const B& it) noexcept(std::is_nothrow_copy_constructible_v<B>)
      : it_(it)
    {}

    //! Copy assignment operator from a mutable iterator.
    constexpr block_const_iter_type& operator=(const B& it) noexcept(std::is_nothrow_copy_assignable_v<B>)
    {
      it_ = it;
      return *this;
    }

    //! Reference to the component at the current iterator position.
    constexpr reference operator*() const { return *it_; }

    //! Compare const iterators for equality and ordering.
    constexpr friend auto operator<=>(const block_const_iter_type& it1,
                                      const B& it2) noexcept
    {
      return it1.baseIterator() <=> it2;
    }

    //! Compare const iterators for equality and ordering.
    constexpr friend auto operator<=>(const B& it1,
                                      const block_const_iter_type& it2) noexcept
    {
      return it1 <=> it2.baseIterator();
    }

  private:
    friend Dune::IteratorFacadeAccess;

    constexpr const B& baseIterator() const noexcept { return it_; }
    constexpr B& baseIterator() noexcept { return it_; }

    //! The underlying mutable block iterator wrapped by this const adapter.
    block_iter_type it_;
  };
#endif

  /**
   * \brief Mutable random-access iterator over matrix rows.
   *
   * Dereferencing yields a \ref row_type view for the current row index.
   * The iterator must not outlive the matrix view it was created from.
   */
  template <std::random_access_iterator B, class I>
  class MatrixView<B, I>::Iterator
      : public Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, row_type, row_type, Dune::ProxyArrowResult<row_type>>
  {
    using Facade = Dune::IteratorFacade<Iterator, std::random_access_iterator_tag, row_type, row_type, Dune::ProxyArrowResult<row_type>>;

    constexpr Iterator(block_iter_type block_iter, pattern_type const* pattern_ptr, size_type row)
      : block_iter_(block_iter), pattern_ptr_(pattern_ptr), row_(row)
    {}

  public:
    using reference = typename Facade::reference;

    /** \brief Dereference to the row view at the current row index. */
    constexpr reference operator*() const {
      auto offset = pattern_ptr_->offset(row_);
      return {block_iter_ + offset, pattern_ptr_->operator[](row_)};
    }

    /** \brief Return the current logical row index. */
    constexpr size_type index() const { return row_; }

  private:
    friend Dune::IteratorFacadeAccess;
    friend MatrixView;

    constexpr const size_type &baseIterator() const noexcept { return row_; }
    constexpr size_type &baseIterator() noexcept { return row_; }

    // View to underlying data
    block_iter_type block_iter_ = {};
    // View to underlying sparsity pattern.
    pattern_type const* pattern_ptr_ = nullptr;
    // The current logical row index of the iterator.
    size_type row_ = 0;
  };

  /**
   * \brief Const random-access iterator over matrix rows.
   *
   * Dereferencing yields a \ref const_row_type view for the current row.
   * The iterator can be constructed or assigned from \ref Iterator.
   */
  template <std::random_access_iterator B, class I>
  class MatrixView<B, I>::ConstIterator
      : public Dune::IteratorFacade<ConstIterator, std::random_access_iterator_tag, const_row_type, const_row_type, Dune::ProxyArrowResult<const_row_type>>
  {
    using Facade = Dune::IteratorFacade<ConstIterator, std::random_access_iterator_tag, const_row_type, const_row_type, Dune::ProxyArrowResult<const_row_type>>;

  public:

    /** \brief Default constructor creating a singular iterator. */
    ConstIterator() noexcept = default;

    /** \brief Construct const iterator from mutable iterator. */
    constexpr ConstIterator(const Iterator& it) noexcept(std::is_nothrow_copy_constructible_v<Iterator>)
      : it_{it}
    {}

    /** \brief Assign from mutable iterator. */
    constexpr ConstIterator &operator=(const Iterator &other) noexcept(std::is_nothrow_copy_assignable_v<Iterator>)
    {
      it_ = other;
      return *this;
    }

    using reference = typename Facade::reference;

    /** \brief Dereference to the const row view at the current row index. */
    constexpr reference operator*() const {
      auto offset = it_.pattern_ptr_->offset(index());
      return {it_.block_iter_ + offset, it_.pattern_ptr_->operator[](index())};
    }

    /** \brief Return the current logical row index. */
    constexpr size_type index() const { return it_.index(); }

  private:
    friend Dune::IteratorFacadeAccess;
    friend Iterator;
    friend MatrixView;

    constexpr const size_type &baseIterator() const noexcept { return it_.baseIterator(); }
    constexpr size_type &baseIterator() noexcept { return it_.baseIterator(); }

    //! The underlying mutable iterator wrapped by this const iterator.
    Iterator it_ = {};
  };

  // MatrixView implementation

  template <std::random_access_iterator B, class I>
  MatrixView<B, I> &MatrixView<B, I>::operator*=(const field_type &k)
  {
    // element-wise scaling of blocks
    for (size_type i = 0; i != pattern().count(); ++i)
      blockIter()[i] *= k;
    return *this;
  }

  template <std::random_access_iterator B, class I>
  MatrixView<B, I> &MatrixView<B, I>::operator/=(const field_type &k)
  {
    // element-wise scaling of blocks
    for (size_type i = 0; i != pattern().count(); ++i)
      blockIter()[i] /= k;
    return *this;
  }

  template <std::random_access_iterator B, class I>
  template <class OtherB, class OtherI>
  MatrixView<B, I> &MatrixView<B, I>::operator+=(const MatrixView<OtherB, OtherI> &b)
  {
    if (pattern_ptr_ == b.pattern_ptr_)
    {
      // element-wise addition of blocks
      for (size_type i = 0; i != pattern().count(); ++i)
        blockIter()[i] += b.blockIter()[i];
    }
    else if (pattern().size() == b.pattern().size() && pattern().range().size() == b.pattern().range().size())
    {
      // row-wise addition of blocks
      for (size_type i = 0; i != pattern().size(); ++i)
        this->operator[](i) += b.operator[](i);
    }
    else
    {
      DUNE_THROW(RangeError, "Matrix sizes do not match!");
    }
    return *this;
  }

  template <std::random_access_iterator B, class I>
  template <class OtherB, class OtherI>
  MatrixView<B, I> &MatrixView<B, I>::operator-=(const MatrixView<OtherB, OtherI> &b)
  {
    if (pattern_ptr_ == b.pattern_ptr_)
    {
      // element-wise subtraction of blocks
      for (size_type i = 0; i != pattern().count(); ++i)
        blockIter()[i] -= b.blockIter()[i];
    }
    else if (pattern().size() == b.pattern().size() && pattern().range().size() == b.pattern().range().size())
    {
      // row-wise subtraction of blocks
      for (size_type i = 0; i != pattern().size(); ++i)
        this->operator[](i) -= b.operator[](i);
    }
    else
    {
      DUNE_THROW(RangeError, "Matrix sizes do not match!");
    }
    return *this;
  }

  template <std::random_access_iterator B, class I>
  MatrixView<B, I> &MatrixView<B, I>::operator=(const field_type &f)
  {
    for (size_type i = 0; i != pattern().count(); ++i)
      blockIter()[i] = f;
    return *this;
  }

  template <std::random_access_iterator B, class I>
  template <class OtherB, class OtherI>
  MatrixView<B, I> &MatrixView<B, I>::axpy(field_type alpha, const MatrixView<OtherB, OtherI> &b)
  {
    if (pattern_ptr_ == b.pattern_ptr_)
    {
      // element-wise axpy addition of blocks
      for (size_type i = 0; i != pattern().count(); ++i)
        Dune::Impl::asMatrix(blockIter()[i]).axpy(alpha, Dune::Impl::asMatrix(b.blockIter()[i]));
    }
    else if (pattern().size() == b.pattern().size() && pattern().range().size() == b.pattern().range().size())
    {
      // row-wise axpy addition of blocks
      for (size_type i = 0; i != pattern().size(); ++i)
        this->operator[](i).axpy(alpha, b.operator[](i));
    }
    else
    {
      DUNE_THROW(RangeError, "Matrix sizes do not match!");
    }
    return *this;
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y, class F, bool ClearRow>
  void MatrixView<B, I>::umvImpl(const X &x, Y &y, F &&op, std::bool_constant<ClearRow>) const
  {
#ifdef DUNE_ISTL_WITH_CHECKING
    if (x.N() != M())
      DUNE_THROW(RangeError,
                 "Size mismatch: M: " << N() << "x" << M() << " x: " << x.N());
    if (y.N() != N())
      DUNE_THROW(RangeError,
                 "Size mismatch: M: " << N() << "x" << M() << " y: " << y.N());
#endif
    for (size_type i = 0; i != N(); ++i) {
      if constexpr (ClearRow)
        y[i] = 0;
      size_type nnz = pattern().offset(i);

      for (size_type j : pattern()[i]) {
        auto &&xj = Dune::Impl::asVector(x[j]);
        auto &&yi = Dune::Impl::asVector(y[i]);
        op(Dune::Impl::asMatrix(blockIter()[nnz]), xj, yi);
        ++nnz;
      }
    }
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y, class F>
  void MatrixView<B, I>::mtvImpl(const X &x, Y &y, F &&op) const
  {
#ifdef DUNE_ISTL_WITH_CHECKING
    if (y.N() != M())
      DUNE_THROW(RangeError,
                 "Size mismatch: M: " << N() << "x" << M() << " y: " << y.N());
    if (x.N() != N())
      DUNE_THROW(RangeError,
                 "Size mismatch: M: " << N() << "x" << M() << " x: " << x.N());
#endif
    for (size_type i = 0; i != N(); ++i) {
      size_type nnz = pattern().offset(i);
      for (size_type j : pattern()[i]) {
        auto &&xi = Dune::Impl::asVector(x[i]);
        auto &&yj = Dune::Impl::asVector(y[j]);
        op(Dune::Impl::asMatrix(blockIter()[nnz]), xi, yj);
        ++nnz;
      }
    }
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::mv(const X &x, Y &y) const
  {
    umvImpl(x, y, [](const auto &aij, const auto &xj, auto &yi){
      aij.umv(xj, yi);
    }, /* clear_row = */ std::bool_constant<true>{});
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::umv(const X &x, Y &y) const
  {
    umvImpl(x, y, [](const auto &aij, const auto &xj, auto &yi){
      aij.umv(xj, yi);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::mmv(const X &x, Y &y) const
  {
    umvImpl(x, y, [](const auto &aij, const auto &xj, auto &yi){
      aij.mmv(xj, yi);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::usmv(const field_type &alpha, const X &x, Y &y) const
  {
    umvImpl(x, y, [&alpha](const auto &aij, const auto &xj, auto &yi){
      aij.usmv(alpha, xj, yi);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::mtv(const X &x, Y &y) const
  {
    using ft = typename FieldTraits<Y>::field_type;
    y = ft(0);
    umtv(x, y);
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::umtv(const X &x, Y &y) const
  {
    mtvImpl(x, y, [](const auto &aij, const auto &xi, auto &yj){
      aij.umtv(xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::mmtv(const X &x, Y &y) const
  {
    mtvImpl(x, y, [](const auto &aij, const auto &xi, auto &yj){
      aij.mmtv(xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::usmtv(const field_type &alpha, const X &x, Y &y) const
  {
    mtvImpl(x, y, [alpha](const auto &aij, const auto &xi, auto &yj){
      aij.usmtv(alpha, xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::umhv(const X &x, Y &y) const
  {
    mtvImpl(x, y, [](const auto &aij, const auto &xi, auto &yj){
      aij.umhv(xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::mmhv(const X &x, Y &y) const
  {
    mtvImpl(x, y, [](const auto &aij, const auto &xi, auto &yj){
      aij.mmhv(xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  template <class X, class Y>
  void MatrixView<B, I>::usmhv(const field_type &alpha, const X &x, Y &y) const
  {
    mtvImpl(x, y, [alpha](const auto &aij, const auto &xi, auto &yj){
      aij.usmhv(alpha, xi, yj);
    });
  }

  template <std::random_access_iterator B, class I>
  auto MatrixView<B, I>::frobenius_norm() const -> typename FieldTraits<field_type>::real_type
  {
    using std::sqrt;
    return sqrt(frobenius_norm2());
  }

  template <std::random_access_iterator B, class I>
  auto MatrixView<B, I>::frobenius_norm2() const -> typename FieldTraits<field_type>::real_type
  {
    using real_type = typename FieldTraits<field_type>::real_type;
    real_type sum = 0;
    for (auto const &a_i : *this)
      for (auto const &a_ij : a_i)
        sum += Dune::Impl::asMatrix(a_ij).frobenius_norm2();
    return sum;
  }

  template <std::random_access_iterator B, class I>
  auto MatrixView<B, I>::infinity_norm() const -> typename FieldTraits<field_type>::real_type
  {
    using real_type = typename FieldTraits<field_type>::real_type;
    constexpr bool hasNaN = std::numeric_limits<real_type>::has_quiet_NaN || std::numeric_limits<real_type>::has_signaling_NaN;

    real_type norm = 0;
    real_type markNaN = 1;
    using std::max;

    for (auto const &a_i : *this)
    {
      real_type const a = a_i.infinity_norm();
      norm = max(a, norm);
      if constexpr (hasNaN)
        markNaN += a;
    }
    if constexpr (hasNaN)
    {
      // maintain NaN exception flags that raised the original NaN values.
      std::fenv_t env;
      int fpe = std::fegetenv(&env);
      markNaN = markNaN / markNaN;
      if (fpe != 0)
        std::fesetenv(&env);
    }

    return (markNaN != markNaN) ? markNaN : norm;
  }

  //! infinity norm (maximum of absolute values of entries)
  template <std::random_access_iterator B, class I>
  auto MatrixView<B, I>::infinity_norm_real() const -> typename FieldTraits<field_type>::real_type
  {
    using real_type = typename FieldTraits<field_type>::real_type;
    constexpr bool hasNaN = std::numeric_limits<real_type>::has_quiet_NaN || std::numeric_limits<real_type>::has_signaling_NaN;

    real_type norm = 0;
    real_type markNaN = 1;
    using std::max;

    for (auto const &a_i : *this)
    {
      real_type const a = a_i.infinity_norm_real();
      norm = max(a, norm);
      if constexpr (hasNaN)
        markNaN += a;
    }
    if constexpr (hasNaN)
    {
      // maintain NaN exception flags that raised the original NaN values.
      std::fenv_t env;
      int fpe = std::fegetenv(&env);
      markNaN = markNaN / markNaN;
      if (fpe != 0)
        std::fesetenv(&env);
    }

    return (markNaN != markNaN) ? markNaN : norm;
  }

} // namespace Dune::Experimental

#endif // DUNE_ISTL_EXPERIMENTAL_MATRIXVIEW_HH
