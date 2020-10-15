// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BCCSMATRIX_HH
#define DUNE_ISTL_BCCSMATRIX_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>

namespace Dune
{
  /**
   * @brief Inititializer for the ColCompMatrix
   * as needed by OverlappingSchwarz
   * @tparam M the matrix type
   * @tparam I the internal index type
   */
  template<class M, class I = int>
  class ColCompMatrixInitializer;

  /**
   * @brief Utility class for converting an ISTL Matrix into a column-compressed matrix
   * @tparam M The matrix type
   * @tparam I the internal index type
   */
  template<class Mat, class I = int>
  class ColCompMatrix
  {
    friend class ColCompMatrixInitializer<Mat, I>;

    using B = typename Mat::field_type;

  public:
    /** @brief The type of the matrix to convert. */
    using Matrix = Mat;

    typedef typename Matrix::size_type size_type;

    using Index = I;

    /**
     * @brief Constructor that initializes the data.
     * @param mat The matrix to convert.
     */
    explicit ColCompMatrix(const Matrix& mat);

    /** \brief Default constructor
     */
    ColCompMatrix()
    : N_(0), M_(0), Nnz_(0), values(0), rowindex(0), colstart(0)
    {}

    /** @brief Destructor */
    virtual ~ColCompMatrix()
    {
      if(N_+M_+Nnz_!=0)
        free();
    }

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
      // TODO: The following code assumes that the blocks are dense
      // and that they all have the same dimensions.
      typename Matrix::block_type dummy;
      const auto n = MatrixDimension<typename Matrix::block_type>::rowdim(dummy);
      const auto m = MatrixDimension<typename Matrix::block_type>::coldim(dummy);
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

    Index* getRowIndex() const
    {
      return rowindex;
    }

    Index* getColStart() const
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
    size_type N_, M_, Nnz_;
    B* values;
    Index* rowindex;
    Index* colstart;
  };

  template<class Mat, class I>
  ColCompMatrix<Mat, I>
  ::ColCompMatrix(const Matrix& mat)
  {
    // WARNING: This assumes that all blocks are dense and identical
    typename Matrix::block_type dummy;
    const auto n = MatrixDimension<typename Matrix::block_type>::rowdim(dummy);
    const auto m = MatrixDimension<typename Matrix::block_type>::coldim(dummy);
    N_ = n*mat.N();
    M_ = m*mat.N();
    Nnz_ = n*m*mat.nonzeroes();
  }

  template<class Mat, class I>
  ColCompMatrix<Mat, I>&
  ColCompMatrix<Mat, I>::operator=(const Matrix& mat)
  {
    if(N_+M_+Nnz_!=0)
      free();
    setMatrix(mat);
    return *this;
  }

  template<class Mat, class I>
  ColCompMatrix<Mat, I>&
  ColCompMatrix<Mat, I>::operator=(const ColCompMatrix& mat)
  {
    if(N_+M_+Nnz_!=0)
      free();
    N_=mat.N_;
    M_=mat.M_;
    Nnz_= mat.Nnz_;
    if(M_>0) {
      colstart=new size_type[M_+1];
      for(size_type i=0; i<=M_; ++i)
        colstart[i]=mat.colstart[i];
    }

    if(Nnz_>0) {
      values = new B[Nnz_];
      rowindex = new size_type[Nnz_];

      for(size_type i=0; i<Nnz_; ++i)
        values[i]=mat.values[i];

      for(size_type i=0; i<Nnz_; ++i)
        rowindex[i]=mat.rowindex[i];
    }
    return *this;
  }

  template<class Matrix>
  class MatrixRowSet;

  template<class Matrix, class Container>
  class MatrixRowSubset;

  template<class Mat, class I>
  void ColCompMatrix<Mat, I>
  ::setMatrix(const Matrix& mat)
  {
    N_=MatrixDimension<Mat>::rowdim(mat);
    M_=MatrixDimension<Mat>::coldim(mat);
    ColCompMatrixInitializer<Mat, I> initializer(*this);

    copyToColCompMatrix(initializer, MatrixRowSet<Matrix>(mat));
  }

  template<class Mat, class I>
  void ColCompMatrix<Mat, I>
  ::setMatrix(const Matrix& mat, const std::set<std::size_t>& mrs)
  {
    if(N_+M_+Nnz_!=0)
      free();

    N_=mrs.size()*MatrixDimension<Mat>::rowdim(mat) / mat.N();
    M_=mrs.size()*MatrixDimension<Mat>::coldim(mat) / mat.M();
    ColCompMatrixInitializer<Mat, I> initializer(*this);

    copyToColCompMatrix(initializer, MatrixRowSubset<Mat,std::set<std::size_t> >(mat,mrs));
  }

  template<class Mat, class I>
  void ColCompMatrix<Mat, I>::free()
  {
    delete[] values;
    delete[] rowindex;
    delete[] colstart;
    N_ = 0;
    M_ = 0;
    Nnz_ = 0;
  }

}
#endif
