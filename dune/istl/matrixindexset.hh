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

  namespace Impl {

    /**
     * @class RowIndexSet
     * @brief Manages a set of row indices with efficient insertion and lookup operations.
     *
     * This class provides a flexible and efficient way to store and manage
     * row indices, supporting both vector and set storage types. It automatically
     * switches between these storage types based on the number of indices to
     * optimize performance for different use cases.
     *
     * @details
     * The `RowIndexSet` class uses a `std::variant` to store row indices either as
     * a `std::vector` or a `std::set`. The vector storage is used initially for
     * its cache-friendly properties and efficient iteration, while the set storage
     * is used when the number of indices exceeds a specified threshold
     * (`maxVectorSize`), providing efficient insertion and lookup operations.
     *
     * This class is not thread safe in the sense that inserting several column
     * indices leads to a data race.
     */
    class RowIndexSet {
      using Index = std::uint_least32_t;
      class Vector : public std::vector<Index> {
        // store max size within class so that variant uses less memory
        size_type maxVectorSize_;
        friend class RowIndexSet;
      };

      //! \brief Maximum vector size for a given vector.
      static Index getMaxVectorSize(const RowIndexSet::Vector& v) {
        return v.maxVectorSize_;
      }

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
       * @brief Constructs a RowIndexSet with a specified maximum vector size.
       *
       * @param maxVectorSize The maximum number of indices to store in a
       * vector before switching to set storage.
       */
      RowIndexSet(size_type maxVectorSize = defaultMaxVectorSize) : storage_{Vector()}
      {
        std::get<Vector>(storage_).maxVectorSize_ = maxVectorSize;
      }

      /**
       * @brief Inserts a column index into the row index set.
       *
       * This function inserts a column index into the row index set, automatically
       * switching to set storage if the maximum vector size is exceeded.
       *
       * @param col The column index to insert.
       */
      void insert(size_type col){
        std::visit(Dune::overload(
          // If row is stored as set, call insert directly
          [&](std::set<size_type>& set) {
            set.insert(col);
          },
          // If row is stored as vector only insert directly
          // if maxVectorSize_ is not reached. Otherwise switch
          // to set storage first.
          [&](Vector& sortedVector) {
            auto it = std::lower_bound(sortedVector.cbegin(), sortedVector.cend(), col);
            if (it == sortedVector.cend() or (*it != col)) {
              if (sortedVector.size() < getMaxVectorSize(sortedVector)) {
                sortedVector.insert(it, col);
              } else {
                std::set<size_type> set(sortedVector.cbegin(), sortedVector.cend());
                set.insert(col);
                storage_ = std::move(set);
              }
            }
          }
        ), storage_);
      }

      /**
       * @brief Checks if the row index set contains a specific column index.
       *
       * @param col The column index to check.
       * @return True if the column index is present in the row index set, false otherwise.
       */
      bool contains(const Index& col) const {
        return std::visit(Dune::overload(
          [&](const std::set<Index>& set) {
            return set.contains(col);
          },
          [&](const std::vector<Index>& sortedVector) {
            return std::binary_search(sortedVector.cbegin(), sortedVector.cend(), col);
          }
        ), storage_);
      }

      /**
       * @brief Returns the current storage of row indices.
       *
       * @return A constant reference to the storage variant containing the row indices.
       */
      const auto& storage() const {
        return storage_;
      }

      /**
       * @brief Returns the number of row indices in the set.
       *
       * @return The number of row indices.
       */
      size_type size() const {
        return std::visit([&](const auto& rowIndices) {
          return rowIndices.size();
        }, storage_);
      }

      /**
       * @brief Clears all row indices from the set.
       */
      void clear() {
        std::visit([&](auto& rowIndices) {
          rowIndices.clear();
        }, storage_);
      }

    private:
      std::variant<Vector, std::set<Index>> storage_;
    };
  }


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
  public:
    using size_type = typename Impl::RowIndexSet::size_type;

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
    static constexpr size_type defaultMaxVectorSize = Impl::RowIndexSet::defaultMaxVectorSize;

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
      indices_.resize(rows_, Impl::RowIndexSet(maxVectorSize_));
    }

    /** \brief Reset the size of an index set */
    void resize(size_type rows, size_type cols) {
      rows_ = rows;
      cols_ = cols;
      indices_.resize(rows_, Impl::RowIndexSet(maxVectorSize_));
    }

    /**
     * \brief Add an index to the index set
     *
     * It is safe to call add(row, col) for different rows in concurrent threads,
     * but it is not safe to do concurrent calls for the same row, even for different
     * columns.
     */
    void add(size_type row, size_type col) {
      indices_[row].insert(col);
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
      return indices_[row].storage();
    }

    /** \brief Return the number of entries in a given row */
    size_type rowsize(size_type row) const {
      return indices_[row].size();
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
        }, indices_[row].storage());
      }

      matrix.endindices();

    }

  private:

    std::vector<Impl::RowIndexSet> indices_;

    size_type rows_, cols_;
    size_type maxVectorSize_;

  };


} // end namespace Dune

#endif
