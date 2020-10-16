// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BCCSMATRIX_HH
#define DUNE_ISTL_BCCSMATRIX_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>

namespace Dune
{
  template<class M, class I = int>
  class ColCompMatrixInitializer;

  /**
   * @brief A block matrix with compressed-column storage
   *
   * @tparam M The matrix type
   * @tparam I the internal index type
   */
  template<class Mat, class I = int>
  class BCCSMatrix
  {
    friend class ColCompMatrixInitializer<Mat, I>;

    using B = typename Mat::field_type;

  public:
    /** @brief The type of the matrix to convert. */
    using Matrix = Mat;

    typedef typename Matrix::size_type size_type;

    using Index = I;

    /** \brief Default constructor
     */
    BCCSMatrix()
    : N_(0), M_(0), Nnz_(0), values(0), rowindex(0), colstart(0)
    {}

    /** @brief Destructor */
    ~BCCSMatrix()
    {
      if(N_+M_+Nnz_!=0)
        free();
    }

    /** \brief Set matrix size */
    void setSize(size_type rows, size_type columns)
    {
      N_ = rows;
      M_ = columns;
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

    /** \brief Assignment operator */
    BCCSMatrix& operator=(const BCCSMatrix& mat)
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

    /** @brief free allocated space. */
    virtual void free()
    {
      delete[] values;
      delete[] rowindex;
      delete[] colstart;
      N_ = 0;
      M_ = 0;
      Nnz_ = 0;
    }

  public:
    size_type N_, M_, Nnz_;
    B* values;
    Index* rowindex;
    Index* colstart;
  };

}
#endif
