// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_MATRIX_INDEX_SET_HH
#define DUNE_MATRIX_INDEX_SET_HH

#include <vector>
#include <set>

namespace Dune {


  /** \brief Stores the nonzero entries in a sparse matrix */
  class MatrixIndexSet
  {

  public:

    /** \brief Default constructor */
    MatrixIndexSet() : rows_(0), cols_(0)
    {}

    /** \brief Constructor setting the matrix size */
    MatrixIndexSet(int rows, int cols) : rows_(rows), cols_(cols) {
      indices_.resize(rows_);
    }

    /** \brief Reset the size of an index set */
    void resize(int rows, int cols) {
      rows_ = rows;
      cols_ = cols;
      indices_.resize(rows_);
    }

    /** \brief Add an index to the index set */
    void add(int i, int j) {
      indices_[i].insert(j);
    }

    /** \brief Return the number of entries */
    int size() const {
      int entries = 0;
      for (int i=0; i<rows_; i++)
        entries += indices_[i].size();

      return entries;
    }

    /** \brief Return the number of rows */
    int rows() const {return rows_;}


    /** \brief Return the number of entries in a given row */
    int rowsize(int row) const {return indices_[row].size();}

    /** \brief Import all nonzero entries of a sparse matrix into the index set
        \param MatrixType Needs to be BCRSMatrix<...>
        \param matrix reference to the MatrixType object
     */
    template <class MatrixType>
    void import(const MatrixType& m, int rowOffset=0, int colOffset=0) {

      typedef typename MatrixType::row_type RowType;
      typedef typename RowType::ConstIterator ColumnIterator;

      for (int rowIdx=0; rowIdx<m.N(); rowIdx++) {

        const RowType& row = m[rowIdx];

        ColumnIterator cIt    = row.begin();
        ColumnIterator cEndIt = row.end();

        for(; cIt!=cEndIt; ++cIt)
          add(rowIdx+rowOffset, cIt.index()+colOffset);

      }

    }

    /** \brief Initializes a BCRSMatrix with the indices contained
        in this MatrixIndexSet
        \param MatrixType Needs to be BCRSMatrix<...>
        \param matrix reference to the MatrixType object
     */
    template <class MatrixType>
    void exportIdx(MatrixType& matrix) const {

      matrix.setSize(rows_, cols_);
      matrix.setBuildMode(MatrixType::random);

      for (int i=0; i<rows_; i++)
        matrix.setrowsize(i, indices_[i].size());

      matrix.endrowsizes();

      for (int i=0; i<rows_; i++) {

        typename std::set<unsigned int>::iterator it = indices_[i].begin();
        for (; it!=indices_[i].end(); ++it)
          matrix.addindex(i, *it);

      }

      matrix.endindices();

    }

  private:

    std::vector<std::set<unsigned int> > indices_;

    int rows_, cols_;

  };


} // end namespace Dune

#endif
