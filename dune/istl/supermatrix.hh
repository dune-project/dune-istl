// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SUPERMATRIX_HH
#define DUNE_ISTL_SUPERMATRIX_HH

#if HAVE_SUPERLU

#include "bcrsmatrix.hh"
#include "bvector.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>
#include <limits>

#include <dune/istl/bccsmatrixinitializer.hh>

#include "superlufunctions.hh"

namespace Dune
{

  template<class T>
  struct SuperMatrixCreateSparseChooser
  {};

  template<class T>
  struct SuperMatrixPrinter
  {};

#if __has_include("slu_sdefs.h")
  template<>
  struct SuperMatrixCreateSparseChooser<float>
  {
    static void create(SuperMatrix *mat, int n, int m, int offset,
                       float *values, int *rowindex, int* colindex,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      sCreate_CompCol_Matrix(mat, n, m, offset, values, rowindex, colindex,
                             stype, dtype, mtype);
    }
  };

  template<>
  struct SuperMatrixPrinter<float>
  {
    static void print(char* name, SuperMatrix* mat)
    {
      sPrint_CompCol_Matrix(name, mat);
    }
  };
#endif

#if __has_include("slu_ddefs.h")
  template<>
  struct SuperMatrixCreateSparseChooser<double>
  {
    static void create(SuperMatrix *mat, int n, int m, int offset,
                       double *values, int *rowindex, int* colindex,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      dCreate_CompCol_Matrix(mat, n, m, offset, values, rowindex, colindex,
                             stype, dtype, mtype);
    }
  };

  template<>
  struct SuperMatrixPrinter<double>
  {
    static void print(char* name, SuperMatrix* mat)
    {
      dPrint_CompCol_Matrix(name, mat);
    }
  };
#endif

#if __has_include("slu_cdefs.h")
  template<>
  struct SuperMatrixCreateSparseChooser<std::complex<float> >
  {
    static void create(SuperMatrix *mat, int n, int m, int offset,
                       std::complex<float> *values, int *rowindex, int* colindex,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      cCreate_CompCol_Matrix(mat, n, m, offset, reinterpret_cast< ::complex*>(values),
                             rowindex, colindex, stype, dtype, mtype);
    }
  };

  template<>
  struct SuperMatrixPrinter<std::complex<float> >
  {
    static void print(char* name, SuperMatrix* mat)
    {
      cPrint_CompCol_Matrix(name, mat);
    }
  };
#endif

#if __has_include("slu_zdefs.h")
  template<>
  struct SuperMatrixCreateSparseChooser<std::complex<double> >
  {
    static void create(SuperMatrix *mat, int n, int m, int offset,
                       std::complex<double> *values, int *rowindex, int* colindex,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      zCreate_CompCol_Matrix(mat, n, m, offset, reinterpret_cast<doublecomplex*>(values),
                             rowindex, colindex, stype, dtype, mtype);
    }
  };

  template<>
  struct SuperMatrixPrinter<std::complex<double> >
  {
    static void print(char* name, SuperMatrix* mat)
    {
      zPrint_CompCol_Matrix(name, mat);
    }
  };
#endif

  template<class T>
  struct BaseGetSuperLUType
  {
    static const Dtype_t type;
  };

  template<class T>
  struct GetSuperLUType
  {};

  template<class T>
  const Dtype_t BaseGetSuperLUType<T>::type =
    std::is_same<T,float>::value ? SLU_S :
    (  std::is_same<T,std::complex<double> >::value ? SLU_Z :
       ( std::is_same<T,std::complex<float> >::value ? SLU_C : SLU_D ));

  template<>
  struct GetSuperLUType<double>
    : public BaseGetSuperLUType<double>
  {
    typedef double float_type;
  };

  template<>
  struct GetSuperLUType<float>
    : public BaseGetSuperLUType<float>
  {
    typedef float float_type;
  };

  template<>
  struct GetSuperLUType<std::complex<double> >
    : public BaseGetSuperLUType<std::complex<double> >
  {
    typedef double float_type;
  };

  template<>
  struct GetSuperLUType<std::complex<float> >
    : public BaseGetSuperLUType<std::complex<float> >
  {
    typedef float float_type;

  };

  /**
   * @brief Utility class for converting an ISTL Matrix
   * into a SuperLU Matrix.
   */
  template<class M>
  struct SuperLUMatrix
  {};

  template<class M>
  struct SuperMatrixInitializer
  {};

  template<class T>
  class SuperLU;

  template<class M, class X, class TM, class TD, class T1>
  class SeqOverlappingSchwarz;

  template<class T, bool flag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /**
   * @brief Converter for BCRSMatrix to SuperLU Matrix.
   */
  template<class B, class TA>
  class SuperLUMatrix<BCRSMatrix<B,TA> >
    : public ISTL::Impl::BCCSMatrix<typename BCRSMatrix<B,TA>::field_type, int>
  {
    template<class M, class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;
    friend struct SuperMatrixInitializer<BCRSMatrix<B,TA> >;
  public:
    /** @brief The type of the matrix to convert. */
    typedef BCRSMatrix<B,TA> Matrix;

    friend struct SeqOverlappingSchwarzAssemblerHelper<SuperLU<Matrix>, true>;

    typedef typename Matrix::size_type size_type;

    /**
     * @brief Constructor that initializes the data.
     * @param mat The matrix to convert.
     */
    explicit SuperLUMatrix(const Matrix& mat) : ISTL::Impl::BCCSMatrix<BCRSMatrix<B,TA>, int>(mat)
    {}

    SuperLUMatrix() : ISTL::Impl::BCCSMatrix<typename BCRSMatrix<B,TA>::field_type, int>()
    {}

    /** @brief Destructor */
    virtual ~SuperLUMatrix()
    {
      if (this->N_+this->M_*this->Nnz_ != 0)
        free();
    }

    /** @brief Cast to a SuperLU Matrix */
    operator SuperMatrix&()
    {
      return A;
    }

    /** @brief Cast to a SuperLU Matrix */
    operator const SuperMatrix&() const
    {
      return A;
    }

    SuperLUMatrix<BCRSMatrix<B,TA> >& operator=(const BCRSMatrix<B,TA>& mat)
    {
      if (this->N_ + this->M_ + this->Nnz_!=0)
        free();

      using Matrix = BCRSMatrix<B,TA>;
      this->N_ = MatrixDimension<Matrix>::rowdim(mat);
      this->M_ = MatrixDimension<Matrix>::coldim(mat);
      ISTL::Impl::BCCSMatrixInitializer<Matrix, int> initializer(*this);

      copyToBCCSMatrix(initializer, mat);

      SuperMatrixCreateSparseChooser<typename Matrix::field_type>
           ::create(&A, this->N_, this->M_, this->colstart[this->N_],
             this->values,this->rowindex, this->colstart, SLU_NC,
             static_cast<Dtype_t>(GetSuperLUType<typename Matrix::field_type>::type), SLU_GE);
      return *this;
    }

    SuperLUMatrix<BCRSMatrix<B,TA> >& operator=(const SuperLUMatrix <BCRSMatrix<B,TA> >& mat)
    {
      if (this->N_ + this->M_ + this->Nnz_!=0)
        free();

      using Matrix = BCRSMatrix<B,TA>;
      this->N_ = MatrixDimension<Matrix>::rowdim(mat);
      this->M_ = MatrixDimension<Matrix>::coldim(mat);
      ISTL::Impl::BCCSMatrixInitializer<Matrix, int> initializer(*this);

      copyToBCCSMatrix(initializer, mat);

      SuperMatrixCreateSparseChooser<B>
           ::create(&A, this->N_, this->M_, this->colstart[this->N_],
             this->values,this->rowindex, this->colstart, SLU_NC,
             static_cast<Dtype_t>(GetSuperLUType<B>::type), SLU_GE);
      return *this;
    }

    /**
     * @brief Initialize data from a given set of matrix rows and columns
     * @tparam The type of the row index set.
     * @param mat the matrix with the values
     * @param mrs The set of row (and column) indices to represent
     */
    virtual void setMatrix(const Matrix& mat, const std::set<std::size_t>& mrs)
    {
      if(this->N_+this->M_+this->Nnz_!=0)
        free();
      this->N_=mrs.size()*MatrixDimension<typename Matrix::block_type>::rowdim(*(mat[0].begin()));
      this->M_=mrs.size()*MatrixDimension<typename Matrix::block_type>::coldim(*(mat[0].begin()));
      SuperMatrixInitializer<Matrix> initializer(*this);

      copyToBCCSMatrix(initializer, ISTL::Impl::MatrixRowSubset<Matrix,std::set<std::size_t> >(mat,mrs));
    }

    /** @brief Initialize data from given matrix. */
    virtual void setMatrix(const Matrix& mat)
    {
      this->N_=MatrixDimension<Matrix>::rowdim(mat);
      this->M_=MatrixDimension<Matrix>::coldim(mat);
      SuperMatrixInitializer<Matrix> initializer(*this);

      copyToBCCSMatrix(initializer, mat);
    }

    /** @brief free allocated space. */
    virtual void free()
    {
      ISTL::Impl::BCCSMatrix<typename BCRSMatrix<B,TA>::field_type, int>::free();
      SUPERLU_FREE(A.Store);
    }
  private:
    SuperMatrix A;
  };

  template<class B, class A>
  class SuperMatrixInitializer<BCRSMatrix<B,A> >
    : public ISTL::Impl::BCCSMatrixInitializer<BCRSMatrix<B,A>, int>
  {
    template<class I, class S, class D>
    friend class OverlappingSchwarzInitializer;
  public:
    typedef BCRSMatrix<B,A> Matrix;
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;

    SuperMatrixInitializer(SuperLUMatrix& lum) : ISTL::Impl::BCCSMatrixInitializer<BCRSMatrix<B,A>, int>(lum)
      ,slumat(&lum)
    {}

    SuperMatrixInitializer() : ISTL::Impl::BCCSMatrixInitializer<BCRSMatrix<B,A>, int>()
    {}

    virtual void createMatrix() const
    {
      ISTL::Impl::BCCSMatrixInitializer<BCRSMatrix<B,A>, int>::createMatrix();
      SuperMatrixCreateSparseChooser<typename Matrix::field_type>
           ::create(&slumat->A, slumat->N_, slumat->M_, slumat->colstart[this->cols],
             slumat->values,slumat->rowindex, slumat->colstart, SLU_NC,
             static_cast<Dtype_t>(GetSuperLUType<typename Matrix::field_type>::type), SLU_GE);
    }
    private:
    SuperLUMatrix* slumat;
  };
}
#endif // HAVE_SUPERLU
#endif
