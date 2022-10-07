// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_TEST_MATRIXTEST_HH
#define DUNE_ISTL_TEST_MATRIXTEST_HH

/** \file
 * \brief Infrastructure for testing the dune-istl matrix interface
 *
 * This file contains various methods that test parts of the dune-istl matrix interface.
 * They only test very general features, i.e., features that every dune-istl matrix should
 * have.
 *
 * At the same time, these tests should help to define what the dune-istl matrix interface
 * actually is.  They may not currently define the entire interface, but they are a lower
 * bound: any feature that is tested here is part of the dune-istl matrix interface.
 */

#include <dune/common/exceptions.hh>
#include <dune/common/test/iteratortest.hh>

namespace Dune
{
  /** \brief Test whether the given type is default- and copy-constructible */
  template <typename Matrix>
  void testMatrixConstructibility()
  {
    static_assert(std::is_default_constructible<Matrix>::value, "Matrix type is not default constructible");
    static_assert(std::is_copy_constructible<Matrix>::value, "Matrix type is not copy constructible");
  }

  template <typename Matrix, typename DomainVector, typename RangeVector>
  void testMatrixVectorProducts(const Matrix& matrix, const DomainVector& domain, const RangeVector& range)
  {
    // Make mutable copies of the vectors
    DomainVector domainMutable = domain;
    RangeVector rangeMutable = range;

    matrix.mv(domain,rangeMutable);
    matrix.umv(domain,rangeMutable);
    matrix.mmv(domain,rangeMutable);
    matrix.usmv(1.0,domain,rangeMutable);
    matrix.mtv(range,domainMutable);
    matrix.umtv(range,domainMutable);
    matrix.mmtv(range,domainMutable);
    matrix.usmtv(1.0,range,domainMutable);
    matrix.umhv(domain,rangeMutable);
    matrix.mmhv(domain,rangeMutable);
    matrix.usmhv(1.0,range,domainMutable);
  }

  /** \brief Test whether a given type implements all the norms required from a dune-istl matrix
   */
  template <typename Matrix>
  void testNorms(const Matrix& m)
  {
    using field_type = typename Matrix::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;

    // frobenius_norm
    static_assert(std::is_same<decltype(m.frobenius_norm()),real_type>::value, "'frobenius_norm' does not return 'real_type'");
    if (m.frobenius_norm() < 0.0)
      DUNE_THROW(RangeError, "'frobenius_norm' returns negative value");

    // frobenius_norm2
    static_assert(std::is_same<decltype(m.frobenius_norm2()),real_type>::value, "'frobenius_norm2' does not return 'real_type'");
    if (m.frobenius_norm2() < 0.0)
      DUNE_THROW(RangeError, "'frobenius_norm2' returns negative value");

    // infinity_norm
    static_assert(std::is_same<decltype(m.infinity_norm()),real_type>::value, "'infinity_norm' does not return 'real_type'");
    if (m.infinity_norm() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm' returns negative value");

    // infinity_norm_real
    static_assert(std::is_same<decltype(m.infinity_norm_real()),real_type>::value, "'infinity_norm_real' does not return 'real_type'");
    if (m.infinity_norm_real() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm_real' returns negative value");
  }

  template <typename Matrix>
  void testVectorSpaceOperations(const Matrix& m)
  {
    // Make a mutable copy of the argument
    Matrix mMutable = m;

    // operator+=
    mMutable += m;
    // operator-=
    mMutable -= m;

    // operator*=
    mMutable *= 0.5;

    // operator/=
    mMutable /= 0.5;

    // axpy -- not sure whether that really is an interface method
    //mMutable.axpy(0.5,m);
  }

}  // namespace Dune

#endif
