// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SUPERMATRIX_HH
#define DUNE_ISTL_SUPERMATRIX_HH

#if HAVE_SUPERLU

#ifndef SUPERLU_NTYPE
#define SUPERLU_NTYPE 1
#endif

#if SUPERLU_NTYPE==0
#include "slu_sdefs.h"
#endif

#if SUPERLU_NTYPE==1
#include "slu_ddefs.h"
#endif

#if SUPERLU_NTYPE==2
#include "slu_cdefs.h"
#endif

#if SUPERLU_NTYPE>=3
#include "slu_zdefs.h"
#endif

#include "bcrsmatrix.hh"
#include "bvector.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>
#include <limits>

#include"colcompmatrix.hh"

namespace Dune
{

  template<class T>
  struct SuperMatrixCreateSparseChooser
  {};

  template<class T>
  struct SuperMatrixPrinter
  {};

#if SUPERLU_NTYPE==0
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

#if SUPERLU_NTYPE==1
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

#if SUPERLU_NTYPE==2
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

#if SUPERLU_NTYPE>=3
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

  /**
   * @brief Converter for BCRSMatrix to SuperLU Matrix.
   */
  template<class B, class TA, int n, int m>
  class SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
    : public ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  {
    template<class M, class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;
    friend struct SuperMatrixInitializer<BCRSMatrix<FieldMatrix<B,n,m>,TA> >;
  public:
    /** @brief The type of the matrix to convert. */
    typedef BCRSMatrix<FieldMatrix<B,n,m>,TA> Matrix;

    friend struct SeqOverlappingSchwarzAssemblerHelper<SuperLU<Matrix>, true>;

    typedef typename Matrix::size_type size_type;

    /**
     * @brief Constructor that initializes the data.
     * @param mat The matrix to convert.
     */
    explicit SuperLUMatrix(const Matrix& mat) : ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >(mat)
    {}

    SuperLUMatrix() : ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >()
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

    SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >& operator=(const BCRSMatrix<FieldMatrix<B,n,m>,TA>& mat)
    {
      this->ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(mat);
      SuperMatrixCreateSparseChooser<B>
           ::create(&A, this->N_, this->M_, this->colstart[this->N_],
             this->values,this->rowindex, this->colstart, SLU_NC,
             static_cast<Dtype_t>(GetSuperLUType<B>::type), SLU_GE);
      return *this;
    }

    SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >& operator=(const SuperLUMatrix <BCRSMatrix<FieldMatrix<B,n,m>,TA> >& mat)
    {
      this->ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(mat);
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
      this->N_=mrs.size()*n;
      this->M_=mrs.size()*m;
      SuperMatrixInitializer<Matrix> initializer(*this);

      copyToColCompMatrix(initializer, MatrixRowSubset<Matrix,std::set<std::size_t> >(mat,mrs));
    }

    /** @brief Initialize data from given matrix. */
    virtual void setMatrix(const Matrix& mat)
    {
      this->N_=n*mat.N();
      this->M_=m*mat.M();
      SuperMatrixInitializer<Matrix> initializer(*this);

      copyToColCompMatrix(initializer, MatrixRowSet<Matrix>(mat));
    }

    /** @brief free allocated space. */
    virtual void free()
    {
      ColCompMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::free();
      SUPERLU_FREE(A.Store);
    }
  private:
    SuperMatrix A;
  };

  template<class T, class A, int n, int m>
  class SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >
    : public ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  {
    template<class I, class S, class D>
    friend class OverlappingSchwarzInitializer;
  public:
    typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;

    SuperMatrixInitializer(SuperLUMatrix& lum) : ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >(lum)
      ,slumat(&lum)
    {}

    SuperMatrixInitializer() : ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >()
    {}

    virtual void createMatrix() const
    {
      ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::createMatrix();
      SuperMatrixCreateSparseChooser<T>
           ::create(&slumat->A, slumat->N_, slumat->M_, slumat->colstart[this->cols],
             slumat->values,slumat->rowindex, slumat->colstart, SLU_NC,
             static_cast<Dtype_t>(GetSuperLUType<T>::type), SLU_GE);
    }
    private:
    SuperLUMatrix* slumat;
  };
}
#endif // HAVE_SUPERLU
#endif
