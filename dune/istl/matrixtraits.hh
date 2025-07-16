// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_MATRIX_TRAITS_HH
#define DUNE_ISTL_MATRIX_TRAITS_HH

#include <memory>

#include <dune/common/concepts/number.hh>

namespace Dune {

// forward declarations
template<class, class> class BCRSMatrix;
template<class, class> class BDMatrix;
template<class, class> class BlockVector;
template<class, class> class BTDMatrix;
template<class, int> class DiagonalMatrix;
template<class> class DynamicMatrix;
template<class, class> class DynamicVector;
template<class, int, int> class FieldMatrix;
template<class, int> class FieldVector;
template<class, class> class Matrix;
template<class, class...> class MultiTypeBlockMatrix;
template<class...> class MultiTypeBlockVector;
template<class, int> class ScaledIdentityMatrix;

/**
 * The traits MatrixTraits defines the `domain_type` and `range_type`
 * based on a given matrix or linear operator type.
 *
 * For a matrix `A in R^{nxm}` is should define the `domain_type` as
 * a vector type comparable to `R^m` and the `range_type` as a vector
 * type comparable to `R^n`. If `A` is blocked, the inner block types
 * should be compatible in domain and range. If `A` is a number type,
 * the domain and range are defined as identical to `A`.
 */
template<class Matrix>
struct MatrixTraits;

/// Specialization for linear operators that already define domain and range type
template<class LO>
  requires requires{ typename LO::domain_type; typename LO::range_type; }
struct MatrixTraits<LO>
{
  using domain_type = typename LO::domain_type;
  using range_type = typename LO::range_type;
};

/// Specialization for number types
template<Concept::Number N>
struct MatrixTraits<N>
{
  using domain_type = N;
  using range_type  = N;
};

/// Specialization for FieldMatrix
template<class T, int n, int m>
struct MatrixTraits<FieldMatrix<T,n,m>>
{
  using domain_type = FieldVector<typename MatrixTraits<T>::domain_type,m>;
  using range_type  = FieldVector<typename MatrixTraits<T>::range_type,n>;
};

/// Specialization for FieldMatrix
template<class T>
struct MatrixTraits<DynamicMatrix<T>>
{
  using domain_type = DynamicVector<typename MatrixTraits<T>::domain_type,
    std::allocator<typename MatrixTraits<T>::domain_type>>;
  using range_type  = DynamicVector<typename MatrixTraits<T>::range_type,
    std::allocator<typename MatrixTraits<T>::range_type>>;
};

/// Specialization for DiagonalMatrix
template<class T, int n>
struct MatrixTraits<DiagonalMatrix<T,n>>
{
  using domain_type = FieldVector<typename MatrixTraits<T>::domain_type,n>;
  using range_type  = FieldVector<typename MatrixTraits<T>::range_type,n>;
};

/// Specialization for ScaledIdentityMatrix
template<class T, int n>
struct MatrixTraits<ScaledIdentityMatrix<T,n>>
{
  using domain_type = FieldVector<typename MatrixTraits<T>::domain_type,n>;
  using range_type  = FieldVector<typename MatrixTraits<T>::range_type,n>;
};

/// Specialization for BCRSMatrix
template<class T, class A>
struct MatrixTraits<BCRSMatrix<T,A>>
{
  // In case of recursive deduction (e.g., BCRSMatrix<FieldMatrix, Allocator<FieldMatrix>>)
  // the allocator needs to be converted to the sub-block allocator type too (e.g.,
  // Allocator<FieldVector>). Note that matrix allocator is assumed to be the same as
  // the domain/range type of allocators

  using domain_type = BlockVector<typename MatrixTraits<T>::domain_type,
    typename std::allocator_traits<A>::template rebind_alloc<typename MatrixTraits<T>::domain_type>>;
  using range_type  = BlockVector<typename MatrixTraits<T>::range_type,
    typename std::allocator_traits<A>::template rebind_alloc<typename MatrixTraits<T>::range_type>>;
};

/// Specialization for BDMatrix: is identical to the BCRSMatrix specialization
template<class T, class A>
struct MatrixTraits<BDMatrix<T,A>>
  : public MatrixTraits<BCRSMatrix<T,A>>
{};

/// Specialization for BTDMatrix: is identical to the BCRSMatrix specialization
template<class T, class A>
struct MatrixTraits<BTDMatrix<T,A>>
  : public MatrixTraits<BCRSMatrix<T,A>>
{};

/// Specialization for Matrix: is identical to the BCRSMatrix specialization
template<class T, class A>
struct MatrixTraits<Matrix<T,A>>
  : public MatrixTraits<BCRSMatrix<T,A>>
{};

namespace Impl
{
  template <class V>
  struct MultiTypeMatrixTraits;

  // to make the `MatrixTraits` work with `MultiTypeBlockMatrix`, we need to add
  // an intermediate step for the rows, which are typically `MultiTypeBlockVector`
  template<class FirstBlock, class... Blocks>
  struct MultiTypeMatrixTraits<MultiTypeBlockVector<FirstBlock, Blocks...>>
  {
    using domain_type = MultiTypeBlockVector<
      typename MatrixTraits<FirstBlock>::domain_type,
      typename MatrixTraits<Blocks>::domain_type...>;
    using range_type  = typename MatrixTraits<FirstBlock>::range_type;
  };

} // end namespace Impl

/// Specialization for `MultiTypeBlockMatrix` with `MultiTypeBlockVector` rows
template<class FirstRow, class... Rows>
struct MatrixTraits<MultiTypeBlockMatrix<FirstRow, Rows...>>
{
  using domain_type = typename Impl::MultiTypeMatrixTraits<FirstRow>::domain_type;
  using range_type  = MultiTypeBlockVector<
    typename Impl::MultiTypeMatrixTraits<FirstRow>::range_type,
    typename Impl::MultiTypeMatrixTraits<Rows>::range_type... >;
};

} // end namespace Dune

#endif // DUNE_ISTL_MATRIX_TRAITS_HH
