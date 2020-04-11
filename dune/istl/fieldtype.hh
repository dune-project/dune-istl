// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_FIELDTYPE_HH
#define DUNE_ISTL_FIELDTYPE_HH

#include <algorithm>
#include <type_traits>

#include <dune/common/indices.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/hybridutilities.hh>

/*!
 * \file
 * \brief Helper alias to determine the field type of a nested block vector/matrix
 */

// forward declaration
namespace Dune {
template<typename... Args>
class MultiTypeBlockVector;
template<typename FirstRow, typename... Args>
class MultiTypeBlockMatrix;
} // end namespace Dune

namespace Dune::Impl {

// end of recursion
template<typename T, bool isNumber = IsNumber<T>{}>
struct FieldType
{ using type = T; };

// continue with T::field_type
template<typename T>
struct FieldType<T, false>
{ using type = typename T::field_type; };

template<typename... Args>
struct FieldType<MultiTypeBlockVector<Args...>, false>
{
  using V = MultiTypeBlockVector<Args...>;
  using Block0 = std::decay_t<decltype(std::declval<V>()[Dune::Indices::_0])>;
  using FieldType0 = typename FieldType<Block0>::type;
  static constexpr bool hasUniqueFieldType = []{
    bool isSameType = true;
    using namespace Dune::Hybrid;
    forEach(integralRange(index_constant<V::size()>()), [&](auto&& i) {
      using FT = typename FieldType<std::decay_t<decltype(std::declval<V>()[i])>>::type;
      isSameType &= std::is_same_v<FieldType0, FT>;
    });
    return isSameType;
  }();
  // is there is no common type fall back to double
  using type = std::conditional_t<hasUniqueFieldType, FieldType0, double>;
};

template<typename FirstRow, typename... Args>
struct FieldType<MultiTypeBlockMatrix<FirstRow, Args...>, false>
{
  using M = MultiTypeBlockMatrix<FirstRow, Args...>;
  using Block00 = std::decay_t<decltype(std::declval<M>()[Dune::Indices::_0][Dune::Indices::_0])>;
  using FieldType00 = typename FieldType<Block00>::type;
  static constexpr bool hasUniqueFieldType = []{
    bool isSameType = true;
    using namespace Dune::Hybrid;
    forEach(integralRange(index_constant<M::N()>()), [&](auto&& i) {
      using namespace Dune::Hybrid; // needed for icc, see issue #31
      forEach(integralRange(index_constant<M::M()>()), [&](auto&& j) {
        using FT = typename FieldType<std::decay_t<decltype(std::declval<M>()[i][j])>>::type;
        isSameType &= std::is_same_v<FieldType00, FT>;
      });
    });
    return isSameType;
  }();
  // is there is no common type fall back to double
  using type = std::conditional_t<hasUniqueFieldType, FieldType00, double>;
};

} // end namespace Dune::Impl

namespace Dune {

template<typename T>
using FieldType = typename Impl::FieldType<T>::type;

} // end namespace Dune

#endif
