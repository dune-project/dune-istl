// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_COLCOMPMATRIX_HH
#define DUNE_ISTL_COLCOMPMATRIX_HH
#include "bcrsmatrix.hh"
#include "bvector.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/unused.hh>
#include <limits>

namespace Dune
{
  /**
   * @brief Provides access to an iterator over all matrix rows.
   *
   * @tparam M The type of the matrix.
   */
  template<class M>
  class MatrixRowSet
  {
  public:
    //! @brief The type of the matrix.
    typedef M Matrix;
    //! @brief The matrix row iterator type.
    typedef typename Matrix::ConstRowIterator const_iterator;

    /**
     * @brief Construct an row set over all matrix rows.
     * @param m The matrix for which we manage the rows.
     */
    MatrixRowSet(const Matrix& m)
      : m_(m)
    {}

    //! @brief Get the row iterator at the first row.
    const_iterator begin() const
    {
      return m_.begin();
    }
    //! @brief Get the row iterator at the end of all rows.
    const_iterator end() const
    {
      return m_.end();
    }
  private:
    const Matrix& m_;
  };

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
   * @brief Utility class for converting an ISTL Matrix into a column-compressed matrix
   * @tparam M the matrix type
   */
  template<class M>
  struct ColCompMatrix
  {};

  /**
   * @brief Inititializer for the ColCompMatrix
   * as needed by OverlappingSchwarz
   * @tparam M the matrix type
   */
  template<class M>
  struct ColCompMatrixInitializer
  {};

  template<class M, class X, class TM, class TD, class T1>
  class SeqOverlappingSchwarz;

  template<class T, bool flag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /**
   * @brief Converter for BCRSMatrix to column-compressed Matrix.
   * specialization for BCRSMatrix
   */
  template<class B, class TA, int n, int m>
  class ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  {
    friend struct ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<B,n,m>,TA> >;

  public:
    /** @brief The type of the matrix to convert. */
    typedef BCRSMatrix<FieldMatrix<B,n,m>,TA> Matrix;
    friend struct SeqOverlappingSchwarzAssemblerHelper<ColCompMatrix<Matrix>, true>;

    typedef typename Matrix::size_type size_type;

    /**
     * @brief Constructor that initializes the data.
     * @param mat The matrix to convert.
     */
    explicit ColCompMatrix(const Matrix& mat);

    ColCompMatrix();

    /** @brief Destructor */
    virtual ~ColCompMatrix();

    /**
     * @brief Get the number of rows.
     * @return  The number of rows.
     */
    size_type N() const
    {
      return N_;
    }

    size_type nnz() const
    {
      return Nnz_/n/m;
    }

    /**
     * @brief Get the number of columns.
     * @return  The number of columns.
     */
    size_type M() const
    {
      return M_;
    }

    B* getValues() const
    {
      return values;
    }

    int* getRowIndex() const
    {
      return rowindex;
    }

    int* getColStart() const
    {
      return colstart;
    }

    ColCompMatrix& operator=(const Matrix& mat);
    ColCompMatrix& operator=(const ColCompMatrix& mat);

    /**
     * @brief Initialize data from a given set of matrix rows and columns
     * @param mat the matrix with the values
     * @param mrs The set of row (and column) indices to remove
     */
    virtual void setMatrix(const Matrix& mat, const std::set<std::size_t>& mrs);
    /** @brief free allocated space. */
    virtual void free();

    /** @brief Initialize data from given matrix. */
    virtual void setMatrix(const Matrix& mat);

  public:
    int N_, M_, Nnz_;
    B* values;
    int* rowindex;
    int* colstart;
  };

  template<class T, class A, int n, int m>
  class ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  {
    template<class I, class S, class D>
    friend class OverlappingSchwarzInitializer;
  public:
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef Dune::ColCompMatrix<Matrix> ColCompMatrix;
    typedef typename Matrix::row_type::const_iterator CIter;
    typedef typename Matrix::size_type size_type;

    ColCompMatrixInitializer(ColCompMatrix& lum);

    ColCompMatrixInitializer();

    virtual ~ColCompMatrixInitializer();

    template<typename Iter>
    void addRowNnz(const Iter& row) const;

    template<typename Iter, typename Set>
    void addRowNnz(const Iter& row, const Set& s) const;

    void allocate();

    template<typename Iter>
    void countEntries(const Iter& row, const CIter& col) const;

    void countEntries(size_type colidx) const;

    void calcColstart() const;

    template<typename Iter>
    void copyValue(const Iter& row, const CIter& col) const;

    void copyValue(const CIter& col, size_type rowindex, size_type colidx) const;

    virtual void createMatrix() const;

  protected:

    void allocateMatrixStorage() const;

    void allocateMarker();

    ColCompMatrix* mat;
    size_type cols;
    mutable size_type *marker;
  };

  template<class T, class A, int n, int m>
  ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::ColCompMatrixInitializer(ColCompMatrix& mat_)
    : mat(&mat_), cols(mat_.M()), marker(0)
  {
    mat->Nnz_=0;
  }

  template<class T, class A, int n, int m>
  ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::ColCompMatrixInitializer()
    : mat(0), cols(0), marker(0)
  {}

  template<class T, class A, int n, int m>
  ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::~ColCompMatrixInitializer()
  {
    if(marker)
      delete[] marker;
  }

  template<class T, class A, int n, int m>
  template<typename Iter>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::addRowNnz(const Iter& row) const
  {
    mat->Nnz_+=row->getsize();
  }

  template<class T, class A, int n, int m>
  template<typename Iter, typename Map>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::addRowNnz(const Iter& row,
                                                                            const Map& indices) const
  {
    typedef typename  Iter::value_type::const_iterator RIter;
    typedef typename Map::const_iterator MIter;
    MIter siter =indices.begin();
    for(RIter entry=row->begin(); entry!=row->end(); ++entry)
    {
      for(; siter!=indices.end() && *siter<entry.index(); ++siter) ;
      if(siter==indices.end())
        break;
      if(*siter==entry.index())
        // index is in subdomain
        ++mat->Nnz_;
    }
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocate()
  {
    allocateMatrixStorage();
    allocateMarker();
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocateMatrixStorage() const
  {
    mat->Nnz_*=n*m;
    // initialize data
    mat->values=new T[mat->Nnz_];
    mat->rowindex=new int[mat->Nnz_];
    mat->colstart=new int[cols+1];
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocateMarker()
  {
    marker = new typename Matrix::size_type[cols];

    for(size_type i=0; i < cols; ++i)
      marker[i]=0;
  }

  template<class T, class A, int n, int m>
  template<typename Iter>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::countEntries(const Iter& row, const CIter& col) const
  {
    DUNE_UNUSED_PARAMETER(row);
    countEntries(col.index());
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::countEntries(size_type colindex) const
  {
    for(size_type i=0; i < m; ++i)
    {
      assert(colindex*m+i<cols);
      marker[colindex*m+i]+=n;
    }
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::calcColstart() const
  {
    mat->colstart[0]=0;
    for(size_type i=0; i < cols; ++i) {
      assert(i<cols);
      mat->colstart[i+1]=mat->colstart[i]+marker[i];
      marker[i]=mat->colstart[i];
    }
  }

  template<class T, class A, int n, int m>
  template<typename Iter>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::copyValue(const Iter& row, const CIter& col) const
  {
    copyValue(col, row.index(), col.index());
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::copyValue(const CIter& col, size_type rowindex, size_type colindex) const
  {
    for(size_type i=0; i<n; i++) {
      for(size_type j=0; j<m; j++) {
        assert(colindex*m+j<cols-1 || (int)marker[colindex*m+j]<mat->colstart[colindex*m+j+1]);
        assert((int)marker[colindex*m+j]<mat->Nnz_);
        mat->rowindex[marker[colindex*m+j]]=rowindex*n+i;
        mat->values[marker[colindex*m+j]]=(*col)[i][j];
        ++marker[colindex*m+j]; // index for next entry in column
      }
    }
  }

  template<class T, class A, int n, int m>
  void ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::createMatrix() const
  {
    delete[] marker;
    marker=0;
  }

  template<class F, class MRS>
  void copyToColCompMatrix(F& initializer, const MRS& mrs)
  {
    typedef typename MRS::const_iterator Iter;
    typedef typename  std::iterator_traits<Iter>::value_type::const_iterator CIter;
    for(Iter row=mrs.begin(); row!= mrs.end(); ++row)
      initializer.addRowNnz(row);

    initializer.allocate();

    for(Iter row=mrs.begin(); row!= mrs.end(); ++row) {

      for(CIter col=row->begin(); col != row->end(); ++col)
        initializer.countEntries(row, col);
    }

    initializer.calcColstart();

    for(Iter row=mrs.begin(); row!= mrs.end(); ++row) {
      for(CIter col=row->begin(); col != row->end(); ++col) {
        initializer.copyValue(row, col);
      }

    }
    initializer.createMatrix();
  }

  template<class F, class M,class S>
  void copyToColCompMatrix(F& initializer, const MatrixRowSubset<M,S>& mrs)
  {
    typedef MatrixRowSubset<M,S> MRS;
    typedef typename MRS::RowIndexSet SIS;
    typedef typename SIS::const_iterator SIter;
    typedef typename MRS::const_iterator Iter;
    typedef typename std::iterator_traits<Iter>::value_type row_type;
    typedef typename row_type::const_iterator CIter;

    // Calculate upper Bound for nonzeros
    for(Iter row=mrs.begin(); row!= mrs.end(); ++row)
      initializer.addRowNnz(row, mrs.rowIndexSet());

    initializer.allocate();

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

#ifndef DOXYGEN

  template<class B, class TA, int n, int m>
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::ColCompMatrix()
    : N_(0), M_(0), Nnz_(0), values(0), rowindex(0), colstart(0)
  {}

  template<class B, class TA, int n, int m>
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::ColCompMatrix(const Matrix& mat)
    : N_(n*mat.N()), M_(m*mat.M()), Nnz_(n*m*mat.nonzeroes())
  {}

  template<class B, class TA, int n, int m>
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >&
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(const Matrix& mat)
  {
    if(N_+M_+Nnz_!=0)
      free();
    setMatrix(mat);
    return *this;
  }

  template<class B, class TA, int n, int m>
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >&
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(const ColCompMatrix& mat)
  {
    if(N_+M_+Nnz_!=0)
      free();
    N_=mat.N_;
    M_=mat.M_;
    Nnz_= mat.Nnz_;
    if(M_>0) {
      colstart=new int[M_+1];
      for(int i=0; i<=M_; ++i)
        colstart[i]=mat.colstart[i];
    }

    if(Nnz_>0) {
      values = new B[Nnz_];
      rowindex = new int[Nnz_];

      for(int i=0; i<Nnz_; ++i)
        values[i]=mat.values[i];

      for(int i=0; i<Nnz_; ++i)
        rowindex[i]=mat.rowindex[i];
    }
    return *this;
  }

  template<class B, class TA, int n, int m>
  void ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::setMatrix(const Matrix& mat)
  {
    N_=n*mat.N();
    M_=m*mat.M();
    ColCompMatrixInitializer<Matrix> initializer(*this);

    copyToColCompMatrix(initializer, MatrixRowSet<Matrix>(mat));
  }

  template<class B, class TA, int n, int m>
  void ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::setMatrix(const Matrix& mat, const std::set<std::size_t>& mrs)
  {
    if(N_+M_+Nnz_!=0)
      free();
    N_=mrs.size()*n;
    M_=mrs.size()*m;
    ColCompMatrixInitializer<Matrix> initializer(*this);

    copyToColCompMatrix(initializer, MatrixRowSubset<Matrix,std::set<std::size_t> >(mat,mrs));
  }

  template<class B, class TA, int n, int m>
  ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::~ColCompMatrix()
  {
    if(N_+M_+Nnz_!=0)
      free();
  }

  template<class B, class TA, int n, int m>
  void ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::free()
  {
    delete[] values;
    delete[] rowindex;
    delete[] colstart;
    N_ = 0;
    M_ = 0;
    Nnz_ = 0;
  }

#endif // DOXYGEN

}
#endif
