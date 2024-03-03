// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIXINDEXSET_HH
#define DUNE_ISTL_MATRIXINDEXSET_HH

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <variant>
#include <vector>

#include <dune/common/overloadset.hh>

namespace Dune {


  /**
   * \brief Stores the nonzero entries for creating a sparse matrix
   *
   * This stores std::set-like container for the column index
   * of each row. A sorted std::vector is used for this container
   * up to a customizable maxVectorSize. If this size is exceeded,
   * storage of the corresponding row is switched to std::set.
   *
   * The default value for maxVectorSize works well and ensures
   * that the slow std::set fallback is only used for very
   * dense rows.
   *
   * This class is thread safe in the sense that concurrent calls
   * to all const methods and furthermore to add(row,col) with different
   * rows in each thread are safe.
   */
  class MatrixIndexSet
  {
    using Index = std::uint_least32_t;

    // A vector that partly mimics a std::set by staying
    // sorted on insert() and having unique values.
    class FlatSet : public std::vector<Index>
    {
      using Base = std::vector<Index>;
    public:
      using Base::Base;
      using Base::begin;
      using Base::end;
      void insert(const Index& value) {
        auto it = std::lower_bound(begin(), end(), value);
        if ((it == end() or (*it != value)))
          Base::insert(it, value);
      }
      bool contains(const Index& value) const {
        return std::binary_search(begin(), end(), value);
      }
    };

    using RowIndexSet = std::variant<FlatSet, std::set<Index>>;

  public:

    using size_type = Index;

    /**
     * \brief Default value for maxVectorSize
     *
     * This was selected after benchmarking for the worst case
     * of reverse insertion of column indices. In many applications
     * this works well. There's no need to use a different value
     * unless you have many dense rows with more than defaultMaxVectorSize
     * nonzero entries. Even in this case defaultMaxVectorSize may work
     * well and a finding a better value may require careful
     * benchmarking.
     */
    static constexpr size_type defaultMaxVectorSize = 2048;

    /**
     * \brief Constructor with custom maxVectorSize
     *
     * \param maxVectorSize Maximal size for stored row vector (default is defaultMaxVectorSize).
     */
    MatrixIndexSet(size_type maxVectorSize=defaultMaxVectorSize) noexcept : rows_(0), cols_(0), maxVectorSize_(maxVectorSize)
    {}

    /**
     * \brief Constructor setting the matrix size
     *
     * \param rows Number of matrix rows
     * \param cols Number of matrix columns
     * \param maxVectorSize Maximal size for stored row vector (default is defaultMaxVectorSize).
     */
    MatrixIndexSet(size_type rows, size_type cols, size_type maxVectorSize=defaultMaxVectorSize) : rows_(rows), cols_(cols), maxVectorSize_(maxVectorSize)
    {
      indices_.resize(rows_, FlatSet());
    }

    /** \brief Reset the size of an index set */
    void resize(size_type rows, size_type cols) {
      rows_ = rows;
      cols_ = cols;
      indices_.resize(rows_, FlatSet());
    }

    /**
     * \brief Add an index to the index set
     *
     * It is safe to call add(row, col) for different rows in concurrent threads,
     * but it is not safe to do concurrent calls for the same row, even for different
     * columns.
     */
    void add(size_type row, size_type col) {
      return std::visit(Dune::overload(
        // If row is stored as set, call insert directly
        [&](std::set<size_type>& set) {
          set.insert(col);
        },
        // If row is stored as vector only insert directly
        // if maxVectorSize_ is not reached. Otherwise switch
        // to set storage first.
        [&](FlatSet& sortedVector) {
          if (sortedVector.size() < maxVectorSize_)
            sortedVector.insert(col);
          else if (not sortedVector.contains(col))
          {
            std::set<size_type> set(sortedVector.begin(), sortedVector.end());
            set.insert(col);
            indices_[row] = std::move(set);
          }
        }
      ), indices_[row]);
    }

    /** \brief Return the number of entries */
    size_type size() const {
      size_type entries = 0;
      for (size_type i=0; i<rows_; i++)
        entries += rowsize(i);
      return entries;
    }

    /** \brief Return the number of rows */
    size_type rows() const {return rows_;}

    /** \brief Return the number of columns */
    size_type cols() const {return cols_;}

    /**
     * \brief Return column indices of entries in given row
     *
     * This returns a range of all column indices
     * that have been added for the given column.
     * Since there are different internal implementations
     * of this range, the result is stored in a std::variant<...>
     * which has to be accessed using `std::visit`.
     */
    const auto& columnIndices(size_type row) const {
      return indices_[row];
    }

    /** \brief Return the number of entries in a given row */
    size_type rowsize(size_type row) const {
      return std::visit([&](const auto& rowIndices) {
        return rowIndices.size();
      }, indices_[row]);
    }

    /** \brief Import all nonzero entries of a sparse matrix into the index set
        \tparam MatrixType Needs to be BCRSMatrix<...>
        \param m reference to the MatrixType object
        \param rowOffset don't write to rows<rowOffset
        \param colOffset don't write to cols<colOffset
     */
    template <class MatrixType>
    void import(const MatrixType& m, size_type rowOffset=0, size_type colOffset=0) {

      typedef typename MatrixType::row_type RowType;
      typedef typename RowType::ConstIterator ColumnIterator;

      for (size_type rowIdx=0; rowIdx<m.N(); rowIdx++) {

        const RowType& row = m[rowIdx];

        ColumnIterator cIt    = row.begin();
        ColumnIterator cEndIt = row.end();

        for(; cIt!=cEndIt; ++cIt)
          add(rowIdx+rowOffset, cIt.index()+colOffset);

      }

    }

    /** \brief Initializes a BCRSMatrix with the indices contained
        in this MatrixIndexSet
        \tparam MatrixType Needs to be BCRSMatrix<...>
        \param matrix reference to the MatrixType object
     */
    template <class MatrixType>
    void exportIdx(MatrixType& matrix) const {

      matrix.setSize(rows_, cols_);
      matrix.setBuildMode(MatrixType::random);

      for (size_type row=0; row<rows_; row++)
        matrix.setrowsize(row, rowsize(row));

      matrix.endrowsizes();

      for (size_type row=0; row<rows_; row++) {
        std::visit([&](const auto& rowIndices) {
          matrix.setIndicesNoSort(row, rowIndices.begin(), rowIndices.end());
        }, indices_[row]);
      }

      matrix.endindices();

    }

  private:

    std::vector<RowIndexSet> indices_;

    size_type rows_, cols_;
    size_type maxVectorSize_;

  };


} // end namespace Dune

#endif
