// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH
#define DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH

#include <cmath>
#include <iostream>
#include <tuple>

#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/std/type_traits.hh>

#include "istlexception.hh"

// forward declaration
namespace Dune {
  template < typename... Args >
  class MultiTypeBlockVector;
}

#include "gsetc.hh"

namespace Dune {

  /**
      @addtogroup ISTL_SPMV
      @{
   */




  /** @addtogroup DenseMatVec
      @{
   */
  template <typename... Args>
  struct FieldTraits< MultiTypeBlockVector<Args...> >
  {
    using field_type = typename MultiTypeBlockVector<Args...>::field_type;
    using real_type  = typename FieldTraits<field_type>::real_type;
  };
  /**
      @}
   */

  /**
      @brief A Vector class to support different block types

      This vector class combines elements of different types known at compile-time.
   */
  template < typename... Args >
  class MultiTypeBlockVector
  : public std::tuple<Args...>
  {
    /** \brief Helper type */
    typedef std::tuple<Args...> TupleType;
  public:

    /** \brief Type used for vector sizes */
    using size_type = std::size_t;

    /**
     * \brief Get the constructors from tuple
     */
    using TupleType::TupleType;

    /**
     * own class' type
     */
    typedef MultiTypeBlockVector<Args...> type;

    /** \brief The type used for scalars
     *
     * Use the `std::common_type_t` of the `Args`' field_type and use `nonesuch` if no implementation of
     * `std::common_type` is provided for the given `field_type` arguments.
     */
    using field_type = Std::detected_t<std::common_type_t, typename FieldTraits< std::decay_t<Args> >::field_type...>;

    // make sure that we have an std::common_type: using an assertion produces a more readable error message
    // than a compiler template instantiation error
    static_assert ( sizeof...(Args) == 0 or not std::is_same_v<field_type, Std::nonesuch>,
        "No std::common_type implemented for the given field_types of the Args. Please provide an implementation and check that a FieldTraits class is present for your type.");


    /** \brief Return the number of non-zero vector entries
     *
     * As this is a dense vector data structure, all entries are non-zero,
     * and hence 'size' returns the same number as 'N'.
     */
    static constexpr size_type size()
    {
      return sizeof...(Args);
    }

    /** \brief Number of elements
     */
    static constexpr size_type N()
    {
      return sizeof...(Args);
    }

    /**
     * number of elements
     *
     * \deprecated Use method <code>N</code> instead.
     *             This will be removed after Dune 2.8.
     */
    [[deprecated("Use method 'N' instead")]]
    int count() const
    {
      return sizeof...(Args);
    }

    /** \brief Number of scalar elements */
    size_type dim() const
    {
      size_type result = 0;
      Hybrid::forEach(std::make_index_sequence<N()>{},
                      [&](auto i){result += std::get<i>(*this).dim();});

      return result;
    }

    /** \brief Random-access operator
     *
     * This method mimicks the behavior of normal vector access with square brackets like, e.g., v[5] = 1.
     * The problem is that the return type is different for each value of the argument in the brackets.
     * Therefore we implement a trick using std::integral_constant.  To access the first entry of
     * a MultiTypeBlockVector named v write
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  std::integral_constant<std::size_t,0> _0;
     *  v[_0] = ...
     * \endcode
     * The name '_0' used here as a static replacement of the integer number zero is arbitrary.
     * Any other variable name can be used.  If you don't like the separate variable, you can writee
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  v[std::integral_constant<std::size_t,0>()] = ...
     * \endcode
     */
    template< size_type index >
    typename std::tuple_element<index,TupleType>::type&
    operator[] ([[maybe_unused]] const std::integral_constant< size_type, index > indexVariable)
    {
      return std::get<index>(*this);
    }

    /** \brief Const random-access operator
     *
     * This is the const version of the random-access operator.  See the non-const version for a full
     * explanation of how to use it.
     */
    template< size_type index >
    const typename std::tuple_element<index,TupleType>::type&
    operator[] ([[maybe_unused]] const std::integral_constant< size_type, index > indexVariable) const
    {
      return std::get<index>(*this);
    }

    /** \brief Assignment operator
     */
    template<typename T>
    void operator= (const T& newval) {
      Dune::Hybrid::forEach(*this, [&](auto&& entry) {
        entry = newval;
      });
    }

    /**
     * operator for MultiTypeBlockVector += MultiTypeBlockVector operations
     */
    void operator+= (const type& newv) {
      using namespace Dune::Hybrid;
      forEach(integralRange(Hybrid::size(*this)), [&](auto&& i) {
        (*this)[i] += newv[i];
      });
    }

    /**
     * operator for MultiTypeBlockVector -= MultiTypeBlockVector operations
     */
    void operator-= (const type& newv) {
      using namespace Dune::Hybrid;
      forEach(integralRange(Hybrid::size(*this)), [&](auto&& i) {
        (*this)[i] -= newv[i];
      });
    }

    /** \brief Multiplication with a scalar */
    template<class T,
             std::enable_if_t< IsNumber<T>::value, int> = 0>
    void operator*= (const T& w) {
      Hybrid::forEach(*this, [&](auto&& entry) {
        entry *= w;
      });
    }

    /** \brief Division by a scalar */
    template<class T,
             std::enable_if_t< IsNumber<T>::value, int> = 0>
    void operator/= (const T& w) {
      Hybrid::forEach(*this, [&](auto&& entry) {
        entry /= w;
      });
    }

    field_type operator* (const type& newv) const {
      using namespace Dune::Hybrid;
      return accumulate(integralRange(Hybrid::size(*this)), field_type(0), [&](auto&& a, auto&& i) {
        return a + (*this)[i]*newv[i];
      });
    }

    field_type dot (const type& newv) const {
      using namespace Dune::Hybrid;
      return accumulate(integralRange(Hybrid::size(*this)), field_type(0), [&](auto&& a, auto&& i) {
        return a + (*this)[i].dot(newv[i]);
      });
    }

    /** \brief Compute the 1-norm
     */
    auto one_norm() const {
      using namespace Dune::Hybrid;
      return accumulate(*this, typename FieldTraits<field_type>::real_type(0), [&](auto&& a, auto&& entry) {
        return a + entry.one_norm();
      });
    }

    /** \brief Compute the simplified 1-norm (uses 1-norm also for complex values)
     */
    auto one_norm_real() const {
      using namespace Dune::Hybrid;
      return accumulate(*this, typename FieldTraits<field_type>::real_type(0), [&](auto&& a, auto&& entry) {
        return a + entry.one_norm_real();
      });
    }

    /** \brief Compute the squared Euclidean norm
     */
    typename FieldTraits<field_type>::real_type two_norm2() const {
      using namespace Dune::Hybrid;
      return accumulate(*this, typename FieldTraits<field_type>::real_type(0), [&](auto&& a, auto&& entry) {
        return a + entry.two_norm2();
      });
    }

    /** \brief Compute the Euclidean norm
     */
    typename FieldTraits<field_type>::real_type two_norm() const {return sqrt(this->two_norm2());}

    /** \brief Compute the maximum norm
     */
    typename FieldTraits<field_type>::real_type infinity_norm() const
    {
      using namespace Dune::Hybrid;
      using std::max;
      using real_type = typename FieldTraits<field_type>::real_type;

      real_type result = 0.0;
      // Compute max norm tracking appearing nan values
      // if the field type supports nan.
      if constexpr (HasNaN<field_type>()) {
        // This variable will preserve any nan value
        real_type nanTracker = 1.0;
        using namespace Dune::Hybrid; // needed for icc, see issue #31
        forEach(*this, [&](auto&& entry) {
          real_type entryNorm = entry.infinity_norm();
          result = max(entryNorm, result);
          nanTracker += entryNorm;
        });
        // Incorporate possible nan value into result
        result *= (nanTracker / nanTracker);
      } else {
        using namespace Dune::Hybrid; // needed for icc, see issue #31
        forEach(*this, [&](auto&& entry) {
          result = max(entry.infinity_norm(), result);
        });
      }
      return result;
    }

    /** \brief Compute the simplified maximum norm (uses 1-norm for complex values)
     */
    typename FieldTraits<field_type>::real_type infinity_norm_real() const
    {
      using namespace Dune::Hybrid;
      using std::max;
      using real_type = typename FieldTraits<field_type>::real_type;

      real_type result = 0.0;
      // Compute max norm tracking appearing nan values
      // if the field type supports nan.
      if constexpr (HasNaN<field_type>()) {
        // This variable will preserve any nan value
        real_type nanTracker = 1.0;
        using namespace Dune::Hybrid; // needed for icc, see issue #31
        forEach(*this, [&](auto&& entry) {
          real_type entryNorm = entry.infinity_norm_real();
          result = max(entryNorm, result);
          nanTracker += entryNorm;
        });
        // Incorporate possible nan value into result
        result *= (nanTracker / nanTracker);
      } else {
        using namespace Dune::Hybrid; // needed for icc, see issue #31
        forEach(*this, [&](auto&& entry) {
          result = max(entry.infinity_norm_real(), result);
        });
      }
      return result;
    }

    /** \brief Axpy operation on this vector (*this += a * y)
     *
     * \tparam Ta Type of the scalar 'a'
     */
    template<typename Ta>
    void axpy (const Ta& a, const type& y) {
      using namespace Dune::Hybrid;
      forEach(integralRange(Hybrid::size(*this)), [&](auto&& i) {
        (*this)[i].axpy(a, y[i]);
      });
    }

  };



  /** \brief Send MultiTypeBlockVector to an outstream
   */
  template <typename... Args>
  std::ostream& operator<< (std::ostream& s, const MultiTypeBlockVector<Args...>& v) {
    using namespace Dune::Hybrid;
    forEach(integralRange(Dune::Hybrid::size(v)), [&](auto&& i) {
      s << "\t(" << i << "):\n" << v[i] << "\n";
    });
    return s;
  }

} // end namespace Dune

namespace std
{
  /** \brief Make std::tuple_element work for MultiTypeBlockVector
   *
   * It derives from std::tuple after all.
   */
  template <size_t i, typename... Args>
  struct tuple_element<i,Dune::MultiTypeBlockVector<Args...> >
  {
    using type = typename std::tuple_element<i, std::tuple<Args...> >::type;
  };
}

#endif
