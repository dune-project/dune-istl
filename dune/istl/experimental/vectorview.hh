// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_EXPERIMENTAL_VECTORVIEW_HH
#define DUNE_ISTL_EXPERIMENTAL_VECTORVIEW_HH

#include "indexedarrayview.hh"

#include "../istlexception.hh"
#include "../blocklevel.hh"

#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/fvector.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/typetraits.hh>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

/*! \file
   \brief Indexed vector view over random-access data with explicit indices.
 */

namespace Dune {
namespace Experimental {

/**
 * \brief Non-owning block vector view with explicit indices.
 *
 * This class extends IndexedArrayView with vector-space operations such as
 * scaled addition and norms. Data ownership remains external.
 *
 * Binary operations with another VectorView iterate over entries of the other
 * vector and expect its indices to form a subset of this view's index set.
 *
 * If DUNE_ISTL_WITH_CHECKING is enabled, index subset violations raise
 * ISTLError. Without checking, violating this precondition is undefined
 * behavior.
 *
 * \internal This class is an implementation detail and should not be used
 * outside dune-istl. In order to use this class, you must derive from it and
 * provide a constructor that initializes the base class.
 * Classes inheriting from this class are expected to specialize FieldTraits to
 * export the field_type and real_type of the block type.
 */
template<std::random_access_iterator B, Concepts::BorrowedOrderedIndexRange R>
class VectorView : public IndexedArrayView<B, R>
{
public:

  //! The type used for the index set
  using index_range_type = typename IndexedArrayView<B, R>::index_range_type;
  //! The type of blocks in the vector
  using block_type = typename IndexedArrayView<B, R>::member_type;
  //! The type used to count elements in the index set
  using size_type = typename IndexedArrayView<B, R>::size_type;
  //! The type of blocks in the vector
  using value_type = typename IndexedArrayView<B, R>::member_type;
  //! The type used for references to the blocks
  using reference = typename IndexedArrayView<B, R>::reference;
  //! The type used for const references to the blocks
  using const_reference = typename IndexedArrayView<B, R>::const_reference;

  //! Iterator type with mutable access to the underlying blocks
  using Iterator = typename IndexedArrayView<B, R>::iterator;
  //! Iterator type with const access to the underlying blocks
  using ConstIterator = typename IndexedArrayView<B, R>::const_iterator;
  //! The type of the field
  using field_type = typename FieldTraits<block_type>::field_type;
  //! The type of the real part of the field
  using real_type = typename FieldTraits<block_type>::real_type;

  //! Assign all entries to scalar k.
  VectorView& operator=(const field_type& k) noexcept(
    noexcept(std::declval<reference>() = std::declval<const field_type&>()))
  {
    for (auto& x_i : *this)
      x_i = k;
    return *this;
  }

  //! Add entries from y to this vector at indices contained in y.
  //! Requires y's indices to be a subset of this vector's indices.
  template<class OtherB, class OtherR>
  VectorView& operator+=(const VectorView<OtherB, OtherR>& y) noexcept(
    noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
    noexcept(std::declval<reference>() +=
             std::declval<typename VectorView<OtherB, OtherR>::const_reference>()))
  {
    forEachIncludedEntry(y, [](auto& x_i, const auto& y_i) { x_i += y_i; });
    return *this;
  }

  //! Subtract entries from y from this vector at indices contained in y.
  //! Requires y's indices to be a subset of this vector's indices.
  template<class OtherB, class OtherR>
  VectorView& operator-=(const VectorView<OtherB, OtherR>& y) noexcept(
    noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
    noexcept(std::declval<reference>() -=
             std::declval<typename VectorView<OtherB, OtherR>::const_reference>()))
  {
    forEachIncludedEntry(y, [](auto& x_i, const auto& y_i) { x_i -= y_i; });
    return *this;
  }

  //! Scale all entries by k.
  VectorView& operator*=(const field_type& k) noexcept(
    noexcept(std::declval<reference>() *= std::declval<const field_type&>()))
  {
    for (auto& x_i : *this)
      x_i *= k;
    return *this;
  }

  //! Divide all entries by k.
  VectorView& operator/=(const field_type& k) noexcept(
    noexcept(std::declval<reference>() /= std::declval<const field_type&>()))
  {
    for (auto& x_i : *this)
      x_i /= k;
    return *this;
  }

  //! AXPY operation on indices from y: this += a * y.
  //! Requires y's indices to be a subset of this vector's indices.
  template<class OtherB, class OtherR>
  VectorView&
  axpy(const field_type& a, const VectorView<OtherB, OtherR>& y) noexcept(
    noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
    noexcept(Dune::Impl::asVector(std::declval<reference>())
               .axpy(std::declval<const field_type&>(),
                     Dune::Impl::asVector(std::declval<typename VectorView<OtherB, OtherR>::const_reference>()))))
  {
    forEachIncludedEntry(y, [a](block_type& x_i, const auto& y_i) {
      Dune::Impl::asVector(x_i).axpy(a, Dune::Impl::asVector(y_i));
    });
    return *this;
  }

  //! Indefinite vector dot product \f$\left (x^T \cdot y \right)\f$.
  template<class OtherB, class OtherR>
  [[nodiscard]]
  auto operator*(const VectorView<OtherB, OtherR>& y) const noexcept(
      noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
      noexcept(
        Dune::Impl::asVector(std::declval<const_reference>()) *
        Dune::Impl::asVector(
          std::declval<typename VectorView<OtherB, OtherR>::const_reference>()))
      )
  {
    checkSizeMatch(y);
    typedef typename PromotionTraits<
      field_type,
      typename VectorView<OtherB, OtherR>::field_type>::PromotedType
      PromotedType;
    PromotedType sum(0);

    auto itY = std::cbegin(y);
    const auto itEndY = std::cend(y);
    const auto itEndX = std::cend(*this);

    for (auto itX = std::begin(*this); itX != itEndX; ++itX) {
      auto indexX = itX.index();
      while (itY != itEndY && itY.index() < indexX)
        ++itY;
      if (itY != itEndY && itY.index() == indexX)
        sum += Dune::Impl::asVector(*itX) * Dune::Impl::asVector(*itY);
    }

    return sum;
  }

  //! Vector dot product \f$\left (x^H \cdot y \right)\f$
  template<class OtherB, class OtherR>
  [[nodiscard]]
  auto dot(const VectorView<OtherB, OtherR>& y) const noexcept(
    noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
    noexcept(Dune::Impl::asVector(std::declval<const_reference>())
               .dot(Dune::Impl::asVector(
                 std::declval<
                   typename VectorView<OtherB, OtherR>::const_reference>()))))
  {
    checkSizeMatch(y);
    typedef typename PromotionTraits<
      field_type,
      typename VectorView<OtherB, OtherR>::field_type>::PromotedType
      PromotedType;
    PromotedType sum(0);

    auto itY = std::cbegin(y);
    const auto itEndY = std::cend(y);
    const auto itEndX = std::cend(*this);

    for (auto itX = std::begin(*this); itX != itEndX; ++itX) {
      auto indexX = itX.index();
      while (itY != itEndY && itY.index() < indexX)
        ++itY;
      if (itY != itEndY && itY.index() == indexX)
        sum += Dune::Impl::asVector(*itX).dot(Dune::Impl::asVector(*itY));
    }

    return sum;
  }

  //! One norm (sum over absolute values of entries)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type one_norm() const noexcept(
    noexcept(Dune::Impl::asVector(std::declval<const_reference>()).one_norm()))
  {
    typename FieldTraits<field_type>::real_type sum = 0;
    for (const auto& x_i : *this)
      sum += Dune::Impl::asVector(x_i).one_norm();
    return sum;
  }

  //! Simplified one norm (uses Manhattan norm for complex values)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type one_norm_real() const noexcept(
    noexcept(Dune::Impl::asVector(std::declval<const_reference>()).one_norm_real()))
  {
    typename FieldTraits<field_type>::real_type sum = 0;
    for (const auto& x_i : *this)
      sum += Dune::Impl::asVector(x_i).one_norm_real();
    return sum;
  }

  //! Two norm sqrt(sum over squared values of entries)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type two_norm() const
    noexcept(noexcept(two_norm2()))
  {
    using std::sqrt;
    return sqrt(two_norm2());
  }

  //! Square of the two-norm (the sum over the squared values of the entries)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type two_norm2() const noexcept(
    noexcept(Dune::Impl::asVector(std::declval<const_reference>()).two_norm2()))
  {
    typename FieldTraits<field_type>::real_type sum = 0;
    for (const auto& x_i : *this)
      sum += Dune::Impl::asVector(x_i).two_norm2();
    return sum;
  }

  //! Infinity norm (maximum of absolute values of entries)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type infinity_norm() const noexcept(
    noexcept(Dune::Impl::asVector(std::declval<const_reference>()).infinity_norm()))
  {
    using real_type = typename FieldTraits<field_type>::real_type;
    constexpr bool hasNaN = std::numeric_limits<real_type>::has_quiet_NaN ||
                            std::numeric_limits<real_type>::has_signaling_NaN;

    real_type norm = 0;
    real_type isNaN = 1;
    using std::max;

    for (auto const& x_i : *this) {
      real_type const a = Dune::Impl::asVector(x_i).infinity_norm();
      norm = max(a, norm);
      if constexpr (hasNaN)
        isNaN += a;
    }
    if constexpr (hasNaN) {
      // maintain NaN exception flags that raised the original NaN values.
      std::fenv_t env;
      int fpe = std::fegetenv(&env);
      isNaN = isNaN / isNaN;
      if (fpe != 0)
        std::fesetenv(&env);
    }

    return (isNaN != isNaN) ? isNaN : norm;
  }

  //! Simplified infinity norm (uses Manhattan norm for complex values)
  [[nodiscard]]
  typename FieldTraits<field_type>::real_type infinity_norm_real() const
    noexcept(noexcept(
      Dune::Impl::asVector(std::declval<const_reference>()).infinity_norm_real()))
  {
    using real_type = typename FieldTraits<field_type>::real_type;
    constexpr bool hasNaN = std::numeric_limits<real_type>::has_quiet_NaN ||
                            std::numeric_limits<real_type>::has_signaling_NaN;

    real_type norm = 0;
    real_type isNaN = 1;
    using std::max;

    for (auto const& x_i : *this) {
      real_type const a = Dune::Impl::asVector(x_i).infinity_norm_real();
      norm = max(a, norm);
      if constexpr (hasNaN)
        isNaN += a;
    }
    if constexpr (hasNaN) {
      // maintain NaN exception flags that raised the original NaN values.
      std::fenv_t env;
      int fpe = std::fegetenv(&env);
      isNaN = isNaN / isNaN;
      if (fpe != 0)
        std::fesetenv(&env);
    }

    return (isNaN != isNaN) ? isNaN : norm;
  }

  //! Logical index span of the vector (size of index_set range), not stored entries.
  [[nodiscard]]
  size_type N() const noexcept
  {
    return std::ranges::size(
      OrderedIndexRangeTraits<index_range_type>::range(this->indexSet()));
  }

protected:
  //!Construct a VectorView from the data and index set.
  using IndexedArrayView<B, R>::IndexedArrayView;

private:

  //! Check that the size of \p other matches this vector's size.
#if DUNE_ISTL_WITH_CHECKING
  template<class OtherB, class OtherR>
  void checkSizeMatch(const VectorView<OtherB, OtherR>& other) const
  {
    if (this->N() != other.N())
      DUNE_THROW(ISTLError,
                 "vector size mismatch: " << this->N() << " vs " << other.N());
  }
#else
  template<class OtherB, class OtherR>
  void checkSizeMatch(const VectorView<OtherB, OtherR>&) const noexcept {}
#endif

  /**
   * \brief Apply \p f to matching entries (x_i, y_i) for every index in y.
   *
   * \pre The index set of y is a subset of this vector's index set.
   *
   * In checking builds, a violated subset precondition throws ISTLError.
   * In non-checking builds, violating the precondition is undefined behavior.
   */
  template<class OtherB, class OtherR>
  void forEachIncludedEntry(
    const VectorView<OtherB, OtherR>& y,
    const std::invocable<
      reference,
      typename VectorView<OtherB, OtherR>::const_reference> auto& f) noexcept(
        noexcept(this->checkSizeMatch(std::declval<const VectorView<OtherB, OtherR>&>())) &&
        noexcept(f(std::declval<reference>(), std::declval<typename VectorView<OtherB, OtherR>::const_reference>()))
      )
  {
    checkSizeMatch(y);
    const auto checkIndex = [](const auto& index,
                               const auto& it,
                               const auto& itEnd,
                               const auto& expect_same) {
#if DUNE_ISTL_WITH_CHECKING
      if (it == itEnd || (expect_same && it.index() != index))
        DUNE_THROW(ISTLError,
                   "index " << index << " not in base array index set");
#endif
    };

    auto itX = std::begin(*this);
    const auto itEndX = std::cend(*this);
    const auto itEndY = std::cend(y);
    for (auto itY = std::cbegin(y); itY != itEndY; ++itY) {
      auto indexY = itY.index();
      checkIndex(indexY, itX, itEndX, std::false_type());
      while (itX.index() < indexY) {
        ++itX;
        checkIndex(indexY, itX, itEndX, std::false_type());
      }
      checkIndex(indexY, itX, itEndX, std::true_type());
      f(*itX, *itY);
    }
  }
};

} // namespace Experimental

template<typename B, typename R>
struct FieldTraits< Dune::Experimental::VectorView<B, R> >
  : public FieldTraits<typename Dune::Experimental::VectorView<B, R>::block_type>
{};


} // namespace Dune

#endif // DUNE_ISTL_EXPERIMENTAL_VECTORVIEW_HH
