// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BCCSMATRIX_INITIALIZER_HH
#define DUNE_ISTL_BCCSMATRIX_INITIALIZER_HH

#include <limits>
#include <set>

#include <dune/common/typetraits.hh>
#include <dune/common/scalarmatrixview.hh>

#include <dune/istl/bccsmatrix.hh>

namespace Dune
{
  template<class I, class S, class D>
  class OverlappingSchwarzInitializer;
}

namespace Dune::ISTL::Impl
{
  /**
   * @brief Provides access to an iterator over an arbitrary subset
   * of matrix rows.
   *
   * @tparam M The type of the matrix.
   * @tparam S the type of the set of valid row indices.
   */
  template<class M, class S>
  class MatrixRowSubset
  {
  public:
    /** @brief the type of the matrix class. */
    typedef M Matrix;
    /** @brief the type of the set of valid row indices. */
    typedef S RowIndexSet;

    /**
     * @brief Construct an row set over all matrix rows.
     * @param m The matrix for which we manage the rows.
     * @param s The set of row indices we manage.
     */
    MatrixRowSubset(const Matrix& m, const RowIndexSet& s)
      : m_(m), s_(s)
    {}

    const Matrix& matrix() const
    {
      return m_;
    }

    const RowIndexSet& rowIndexSet() const
    {
      return s_;
    }

    //! @brief The matrix row iterator type.
    class const_iterator
      : public ForwardIteratorFacade<const_iterator, const typename Matrix::row_type>
    {
    public:
      const_iterator(typename Matrix::const_iterator firstRow,
                     typename RowIndexSet::const_iterator pos)
        : firstRow_(firstRow), pos_(pos)
      {}


      const typename Matrix::row_type& dereference() const
      {
        return *(firstRow_+ *pos_);
      }
      bool equals(const const_iterator& o) const
      {
        return pos_==o.pos_;
      }
      void increment()
      {
        ++pos_;
      }
      typename RowIndexSet::value_type index() const
      {
        return *pos_;
      }

    private:
      //! @brief Iterator pointing to the first row of the matrix.
      typename Matrix::const_iterator firstRow_;
      //! @brief Iterator pointing to the current row index.
      typename RowIndexSet::const_iterator pos_;
    };

    //! @brief Get the row iterator at the first row.
    const_iterator begin() const
    {
      return const_iterator(m_.begin(), s_.begin());
    }
    //! @brief Get the row iterator at the end of all rows.
    const_iterator end() const
    {
      return const_iterator(m_.begin(), s_.end());
    }

  private:
    //! @brief The matrix for which we manage the row subset.
    const Matrix& m_;
    //! @brief The set of row indices we manage.
    const RowIndexSet& s_;
  };

  /**
   * @brief Initializer for a BCCSMatrix,
   * as needed by OverlappingSchwarz
   * @tparam M The matrix type
   * @tparam I The type used for row and column indices
   */
  template<class M, class I = typename M::size_type>
  class BCCSMatrixInitializer
  {
    template<class IList, class S, class D>
    friend class Dune::OverlappingSchwarzInitializer;
  public:
    using Matrix = M;
    using Index = I;
    typedef Dune::ISTL::Impl::BCCSMatrix<typename Matrix::field_type, I> OutputMatrix;
    typedef typename Matrix::size_type size_type;

    /** \brief Constructor for dense matrix-valued matrices
     */
    BCCSMatrixInitializer(OutputMatrix& mat_)
    : mat(&mat_), cols(mat_.M())
    {
      if constexpr (Dune::IsNumber<typename M::block_type>::value)
      {
        n = m = 1;
      }
      else
      {
        // WARNING: This assumes that all blocks are dense and identical
        n = M::block_type::rows;
        m = M::block_type::cols;
      }

      mat->Nnz_=0;
    }

    BCCSMatrixInitializer()
    : mat(0), cols(0), n(0), m(0)
    {}

    virtual ~BCCSMatrixInitializer()
    {}

    template<typename Iter>
    void addRowNnz(const Iter& row) const
    {
      mat->Nnz_+=row->getsize();
    }

    template<typename Iter, typename FullMatrixIndex>
    void addRowNnz(const Iter& row, const std::set<FullMatrixIndex>& indices) const
    {
      auto siter =indices.begin();
      for (auto entry=row->begin(); entry!=row->end(); ++entry)
      {
        for(; siter!=indices.end() && *siter<entry.index(); ++siter) ;
        if(siter==indices.end())
          break;
        if(*siter==entry.index())
          // index is in subdomain
          ++mat->Nnz_;
      }
    }

    template<typename Iter, typename SubMatrixIndex>
    void addRowNnz(const Iter& row, const std::vector<SubMatrixIndex>& indices) const
    {
      for (auto entry=row->begin(); entry!=row->end(); ++entry)
        if (indices[entry.index()]!=std::numeric_limits<SubMatrixIndex>::max())
          ++mat->Nnz_;
    }

    void allocate()
    {
      allocateMatrixStorage();
      allocateMarker();
    }

    template<typename Iter, typename CIter>
    void countEntries([[maybe_unused]] const Iter& row, const CIter& col) const
    {
      countEntries(col.index());
    }

    void countEntries(size_type colindex) const
    {
      for(size_type i=0; i < m; ++i)
      {
        assert(colindex*m+i<cols);
        marker[colindex*m+i]+=n;
      }
    }

    void calcColstart() const
    {
      mat->colstart[0]=0;
      for(size_type i=0; i < cols; ++i) {
        assert(i<cols);
        mat->colstart[i+1]=mat->colstart[i]+marker[i];
        marker[i]=mat->colstart[i];
      }
    }

    template<typename Iter, typename CIter>
    void copyValue(const Iter& row, const CIter& col) const
    {
      copyValue(col, row.index(), col.index());
    }

    template<typename CIter>
    void copyValue(const CIter& col, size_type rowindex, size_type colindex) const
    {
      for(size_type i=0; i<n; i++) {
        for(size_type j=0; j<m; j++) {
          assert(colindex*m+j<cols-1 || (size_type)marker[colindex*m+j]<(size_type)mat->colstart[colindex*m+j+1]);
          assert((size_type)marker[colindex*m+j]<mat->Nnz_);
          mat->rowindex[marker[colindex*m+j]]=rowindex*n+i;
          mat->values[marker[colindex*m+j]] = Dune::Impl::asMatrix(*col)[i][j];
          ++marker[colindex*m+j]; // index for next entry in column
        }
      }
    }

    virtual void createMatrix() const
    {
      marker.clear();
    }

  protected:

    void allocateMatrixStorage() const
    {
      mat->Nnz_*=n*m;
      // initialize data
      mat->values=new typename M::field_type[mat->Nnz_];
      mat->rowindex=new I[mat->Nnz_];
      mat->colstart=new I[cols+1];
    }

    void allocateMarker()
    {
      marker.resize(cols);
      std::fill(marker.begin(), marker.end(), 0);
    }

    OutputMatrix* mat;
    size_type cols;

    // Number of rows/columns of the matrix entries
    // (assumed to be scalars or dense matrices)
    size_type n, m;

    mutable std::vector<size_type> marker;
  };

  template<class F, class Matrix>
  void copyToBCCSMatrix(F& initializer, const Matrix& matrix)
  {
    for (auto row=matrix.begin(); row!= matrix.end(); ++row)
      initializer.addRowNnz(row);

    initializer.allocate();

    for (auto row=matrix.begin(); row!= matrix.end(); ++row) {

      for (auto col=row->begin(); col != row->end(); ++col)
        initializer.countEntries(row, col);
    }

    initializer.calcColstart();

    for (auto row=matrix.begin(); row!= matrix.end(); ++row) {
      for (auto col=row->begin(); col != row->end(); ++col) {
        initializer.copyValue(row, col);
      }

    }
    initializer.createMatrix();
  }

  template<class F, class M,class S>
  void copyToBCCSMatrix(F& initializer, const MatrixRowSubset<M,S>& mrs)
  {
    typedef MatrixRowSubset<M,S> MRS;
    typedef typename MRS::RowIndexSet SIS;
    typedef typename SIS::const_iterator SIter;
    typedef typename MRS::const_iterator Iter;
    typedef typename std::iterator_traits<Iter>::value_type row_type;
    typedef typename row_type::const_iterator CIter;

    typedef typename MRS::Matrix::size_type size_type;

    // A vector containing the corresponding indices in
    // the to create submatrix.
    // If an entry is the maximum of size_type then this index will not appear in
    // the submatrix.
    std::vector<size_type> subMatrixIndex(mrs.matrix().N(),
                                          std::numeric_limits<size_type>::max());
    size_type s=0;
    for(SIter index = mrs.rowIndexSet().begin(); index!=mrs.rowIndexSet().end(); ++index)
      subMatrixIndex[*index]=s++;

    // Calculate upper Bound for nonzeros
    for(Iter row=mrs.begin(); row!= mrs.end(); ++row)
      initializer.addRowNnz(row, subMatrixIndex);

    initializer.allocate();

    for(Iter row=mrs.begin(); row!= mrs.end(); ++row)
      for(CIter col=row->begin(); col != row->end(); ++col) {
        if(subMatrixIndex[col.index()]!=std::numeric_limits<size_type>::max())
          // This column is in our subset (use submatrix column index)
          initializer.countEntries(subMatrixIndex[col.index()]);
      }

    initializer.calcColstart();

    for(Iter row=mrs.begin(); row!= mrs.end(); ++row)
      for(CIter col=row->begin(); col != row->end(); ++col) {
        if(subMatrixIndex[col.index()]!=std::numeric_limits<size_type>::max())
          // This value is in our submatrix -> copy (use submatrix indices
          initializer.copyValue(col, subMatrixIndex[row.index()], subMatrixIndex[col.index()]);
      }
    initializer.createMatrix();
  }

}
#endif
