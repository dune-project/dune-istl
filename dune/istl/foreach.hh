// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#pragma once

#include<type_traits>
#include<utility>
#include<cassert>

#include<dune/common/std/type_traits.hh>
#include<dune/common/diagonalmatrix.hh>
#include<dune/common/hybridutilities.hh>
#include<dune/common/indices.hh>

#include<dune/istl/bcrsmatrix.hh>
#include<dune/istl/scaledidmatrix.hh>

namespace Dune{

  namespace Impl {

  // stolen from dune-functions: we call a type "scalar" if it does not support index accessing
  template<class C>
  using StaticIndexAccessConcept = decltype(std::declval<C>()[Dune::Indices::_0]);

  template<class C>
  using IsScalar = std::negation<Dune::Std::is_detected<StaticIndexAccessConcept, std::remove_reference_t<C>>>;

  // Type trait for matrix types that supports
  // - iteration done row-wise
  // - sparse iteration over nonzero entries
  template <class T>
  struct IsRowMajorSparse : std::false_type {};

  // This is supported by the following matrix types:
  template <class B, class A>
  struct IsRowMajorSparse<BCRSMatrix<B,A>> : std::true_type {};

  template <class K, int n>
  struct IsRowMajorSparse<DiagonalMatrix<K,n>> : std::true_type {};

  template <class K, int n>
  struct IsRowMajorSparse<ScaledIdentityMatrix<K,n>> : std::true_type {};


  template <class Matrix>
  auto rows(Matrix const& /*matrix*/, PriorityTag<2>) -> std::integral_constant<std::size_t, Matrix::N()> { return {}; }

  template <class Matrix>
  auto cols(Matrix const& /*matrix*/, PriorityTag<2>) -> std::integral_constant<std::size_t, Matrix::M()> { return {}; }

  template <class Matrix>
  auto rows(Matrix const& matrix, PriorityTag<1>) -> decltype(matrix.N()) { return matrix.N(); }

  template <class Matrix>
  auto cols(Matrix const& matrix, PriorityTag<1>) -> decltype(matrix.M()) { return matrix.M(); }

  template <class Vector>
  auto size(Vector const& /*vector*/, PriorityTag<2>) -> std::integral_constant<std::size_t, Vector::size()> { return {}; }

  template <class Vector>
  auto size(Vector const& vector, PriorityTag<1>) -> decltype(vector.size()) { return vector.size(); }


  } // end namespace Impl

namespace ForEach{

  template <class Matrix>
  auto rows(Matrix const& matrix) { return Impl::rows(matrix, PriorityTag<5>{}); }

  template <class Matrix>
  auto cols(Matrix const& matrix) { return Impl::cols(matrix, PriorityTag<5>{}); }

  template <class Vector>
  auto size(Vector const& vector) { return Impl::size(vector, PriorityTag<5>{}); }

} // namespace ForEach




/** \brief Traverse a blocked vector and call a functor at each scalar entry
 *
 *  The functor `f` is assumed to have the signature
 *
 *    void(auto&& entry, std::size_t offset)
 *
 *  taking a scalar entry and the current flat offset (index)
 *  of this position.
 *
 *  It returns the total number of scalar entries. Similar to `dimension()` for
 *  some DUNE vector types.
 */
template <class Vector, class F>
std::size_t flatVectorForEach(Vector&& vector, F&& f, std::size_t offset = 0)
{
  using V = std::decay_t<Vector>;
  if constexpr( Impl::IsScalar<V>::value )
  {
    f(vector, offset);
    return 1;
  }
  else
  {
    std::size_t idx = 0;
    Hybrid::forEach(Dune::range(ForEach::size(vector)), [&](auto i) {
      idx += flatVectorForEach(vector[i], f, offset + idx);
    });
    return idx;
  }
}


/** \brief Traverse a blocked matrix and call a functor at each scalar entry
 *
 *  The functor `f` is assumed to have the signature
 *
 *    void(auto&& entry, std::size_t rowOffset, std::size_t colOffset)
 *
 *  taking a scalar entry and the current flat offset (index)
 *  of both row and column.
 *
 *  The restrictions on the matrix are:
 *   - well aligned blocks (otherwise there is no sense in the total number of scalar rows/cols)
 *   - all blocks have positive non-zero column / row number
 *   - at least one entry must be present if dynamic matrix types are wrapped within other dynamic matrix types
 *   - if the block size of a sparse matrix is statically known at compile time, the matrix can be empty
 *
 *  The return value is a pair of the total number of scalar rows and columns of the matrix.
 */
template <class Matrix, class F>
std::pair<std::size_t,std::size_t> flatMatrixForEach(Matrix&& matrix, F&& f, std::size_t rowOffset = 0, std::size_t colOffset = 0)
{
  using M = std::decay_t<Matrix>;
  if constexpr ( Impl::IsScalar<M>::value )
  {
    f(matrix,rowOffset,colOffset);
    return {1,1};
  }
  else
  {
    // if M supports the IsRowMajorSparse type trait: iterate just over the nonzero entries and
    // and compute the flat row/col size directly
    if constexpr ( Impl::IsRowMajorSparse<M>::value )
    {
      using Block = std::decay_t<decltype(matrix[0][0])>;

      // find an existing block or at least try to create one
      auto block = [&]{
        for (auto const& row : matrix)
          for (auto const& entry : row)
            return entry;
        return Block{};
      }();

      // compute the scalar size of the block
      auto [blockRows, blockCols] = flatMatrixForEach(block, [](...){});

      // check whether we have valid sized blocks
      assert( ( blockRows!=0 or blockCols!=0 ) and "the block size can't be zero");

      for ( auto rowIt = matrix.begin(); rowIt != matrix.end(); rowIt++ )
      {
        auto&& row = *rowIt;
        auto rowIdx = rowIt.index();
        for ( auto colIt = row.begin(); colIt != row.end(); colIt++ )
        {
          auto&& entry = *colIt;
          auto colIdx = colIt.index();
          auto [ dummyRows, dummyCols ] = flatMatrixForEach(entry, f, rowOffset + rowIdx*blockRows, colOffset + colIdx*blockCols);
          assert( dummyRows == blockRows and dummyCols == blockCols and "we need the same size of each block in this matrix type");
        }
      }

      return { matrix.N()*blockRows, matrix.M()*blockCols };
    }
    // all other matrix types are accessed index-wise with dynamic flat row/col counting
    else
    {
      std::size_t r = 0, c = 0;
      std::size_t nRows, nCols;

      Hybrid::forEach(Dune::range(ForEach::rows(matrix)), [&](auto i) {
        c = 0;
        Hybrid::forEach(Dune::range(ForEach::cols(matrix)), [&](auto j) {
          std::tie(nRows,nCols) = flatMatrixForEach(matrix[i][j], f, rowOffset + r, colOffset + c);
          c += nCols;
        });
        r += nRows;
      });
      return {r,c};
    }
  }
}

} // namespace Dune
