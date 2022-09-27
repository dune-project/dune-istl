// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_TEST_VECTORTEST_HH
#define DUNE_ISTL_TEST_VECTORTEST_HH

/** \file
 * \brief Infrastructure for testing the dune-istl vector interface
 *
 * This file contains various methods that test parts of the dune-istl vector interface.
 * They only test very general features, i.e., features that every dune-istl vector should
 * have.
 *
 * At the same time, these tests should help to define what the dune-istl vector interface
 * actually is.  They may not currently define the entire interface, but they are a lower
 * bound: any feature that is tested here is part of the dune-istl vector interface.
 */

#include <dune/common/exceptions.hh>
#include <dune/common/test/iteratortest.hh>

namespace Dune
{

  /** \brief Test whether a given data type 'Vector' is a random-access container
   *
   * Somewhat self-recursively, this class defines what dune-istls means by 'random-access-container'.
   *
   * Not every dune-istl vector must comply with this: for example, MultiTypeBlockVector is a
   * heterogeneous container, and will get a separate test.
   */
  template <typename Vector>
  void testHomogeneousRandomAccessContainer(const Vector& v)
  {
    // class value_type
    static_assert(std::is_same<typename Vector::value_type, typename Vector::value_type>::value,
                  "Vector does not export 'value_type'");

    // class block_type, must be equal to value_type
    static_assert(std::is_same<typename Vector::block_type, typename Vector::block_type>::value,
                  "Vector does not export 'block_type'");
    static_assert(std::is_same<typename Vector::block_type, typename Vector::value_type>::value,
                  "'block_type' is not equal to 'value_type'");

    // Check whether 'reference' and 'const_reference' are properly exported
    static_assert(std::is_same<typename Vector::reference, typename Vector::reference>::value,
                  "Vector does not export 'reference'");
    static_assert(std::is_same<typename Vector::const_reference, typename Vector::const_reference>::value,
                  "Vector does not export 'const_reference'");

    // class allocator_type
#if 0  // Out-commented, because it is not clear whether vectors with static allocation should have this
    static_assert(std::is_same<typename Vector::allocator_type, typename Vector::allocator_type>::value,
                  "Vector does not export 'allocator_type'");
#endif
    // Iterator / iterator
    static_assert(std::is_same<typename Vector::Iterator, typename Vector::Iterator>::value,
                  "Vector does not export 'Iterator'");
    static_assert(std::is_same<typename Vector::iterator, typename Vector::iterator>::value,
                  "Vector does not export 'iterator'");
    static_assert(std::is_same<typename Vector::Iterator, typename Vector::iterator>::value,
                  "'Iterator' and 'iterator' are not the same type");

    // - is random-access iterator
    Vector vMutable = v;
    auto noop = [](typename Vector::const_reference t){};
    // This is testing the non-const iterators
    testRandomAccessIterator(vMutable.begin(), vMutable.end(), noop);
    // ConstIterator / const_iterator
    static_assert(std::is_same<typename Vector::ConstIterator, typename Vector::ConstIterator>::value,
                  "Vector does not export 'ConstIterator'");
    static_assert(std::is_same<typename Vector::const_iterator, typename Vector::const_iterator>::value,
                  "Vector does not export 'const_iterator'");
    static_assert(std::is_same<typename Vector::ConstIterator, typename Vector::const_iterator>::value,
                  "'ConstIterator' and 'const_iterator' are not the same type");

    // Test reference types
    static_assert(std::is_same<typename Vector::reference, typename Vector::reference>::value,
                  "Vector does not export 'reference'");
    static_assert(std::is_same<typename Vector::const_reference, typename Vector::const_reference>::value,
                  "Vector does not export 'const_reference'");

    // Test the const_iterator
    testRandomAccessIterator(v.begin(), v.end(), noop);

    // Check whether 'size_type' is exported
    static_assert(std::is_same<typename Vector::size_type, typename Vector::size_type>::value,
                  "Vector does not export 'size_type'");

    // size_type must be integral
    static_assert(std::is_integral<typename Vector::size_type>::value, "'size_type' is not integral!");

    // Check whether methods size() and N() exist, and are consistent
    if (v.size() > v.N())
      DUNE_THROW(RangeError, "size() must be less than or equal to N()");

    // Check whether method 'find' works, and is consistent with operator[]
    for (typename Vector::size_type i=0; i<v.N(); ++i)
    {
      // Check whether find(i) returns the correct type
      static_assert(std::is_same<typename Vector::const_iterator,decltype(v.find(i))>::value, "'find const' does not return const_iterator");
      static_assert(std::is_same<typename Vector::iterator,decltype(vMutable.find(i))>::value, "'find' does not return iterator");

      // Check that if 'find' returns something, it corresponds to the i-th entry
      if (v.find(i) != v.end())
        if (v.find(i)-i != v.begin())
          DUNE_THROW(RangeError, "'find(i)' does not return an iterator to the i-th entry");
    }

  }

  /** \brief Test whether the given type is default- and copy-constructible */
  template <typename Vector>
  void testConstructibility()
  {
    static_assert(std::is_default_constructible<Vector>::value, "Vector type is not default constructible");
    static_assert(std::is_copy_constructible<Vector>::value, "Vector type is not copy constructible");
  }

  /** \brief Test whether a given type implements all the norms required from a dune-istl vector
   */
  template <typename Vector>
  void testNorms(const Vector& v)
  {
    using field_type = typename Vector::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;

    // one_norm
    static_assert(std::is_same<decltype(v.one_norm()),real_type>::value, "'one_norm' does not return 'real_type'");
    if (v.one_norm() < 0.0)
      DUNE_THROW(RangeError, "'one_norm' returns negative value");

    // one_norm_real
    static_assert(std::is_same<decltype(v.one_norm_real()),real_type>::value, "'one_norm_real' does not return 'real_type'");
    if (v.one_norm_real() < 0.0)
      DUNE_THROW(RangeError, "'one_norm_real' returns negative value");

    // two_norm
    static_assert(std::is_same<decltype(v.two_norm()),real_type>::value, "'two_norm' does not return 'real_type'");
    if (v.two_norm() < 0.0)
      DUNE_THROW(RangeError, "'two_norm' returns negative value");

    // two_norm2
    static_assert(std::is_same<decltype(v.two_norm2()),real_type>::value, "'two_norm2' does not return 'real_type'");
    if (v.two_norm2() < 0.0)
      DUNE_THROW(RangeError, "'two_norm2' returns negative value");

    // infinity_norm
    static_assert(std::is_same<decltype(v.infinity_norm()),real_type>::value, "'infinity_norm' does not return 'real_type'");
    if (v.infinity_norm() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm' returns negative value");

    // infinity_norm_real
    static_assert(std::is_same<decltype(v.infinity_norm_real()),real_type>::value, "'infinity_norm_real' does not return 'real_type'");
    if (v.infinity_norm_real() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm_real' returns negative value");
  }

  template <typename Vector>
  void testVectorSpaceOperations(const Vector& v)
  {
    // Make a mutable copy of the argument
    Vector vMutable = v;

    // operator+=
    vMutable += v;
    // operator-=
    vMutable -= v;

    // operator*=
    vMutable *= 0.5;

    // operator/=
    vMutable /= 0.5;

    // axpy
    vMutable.axpy(0.5,v);
  }

  /** \brief Test whether the canonical scalar product behaves as it should */
  template <typename Vector>
  void testScalarProduct(const Vector& v)
  {
    using field_type = typename Vector::field_type;

    static_assert(std::is_same<decltype(v*v),field_type>::value, "operator* (vector product) does not return field_type");
  }
}  // namespace Dune

#endif
