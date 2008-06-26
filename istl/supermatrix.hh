// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SUPERLUMATRIX_HH
#define DUNE_SUPERLUMATRIX_HH

#ifdef HAVE_SUPERLU
#ifdef SUPERLU_POST_2005_VERSION
#include "slu_ddefs.h"
#else
#include "dsp_defs.h"
#endif
#include "bcrsmatrix.hh"
#include "bvector.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
namespace Dune
{


  template<class T>
  struct GetSuperLUType
  {};

  template<>
  struct GetSuperLUType<double>
  {
    enum { type = SLU_D};

  };

  template<>
  struct GetSuperLUType<float>
  {
    enum { type = SLU_S};

  };

  template<>
  struct GetSuperLUType<std::complex<double> >
  {
    enum { type = SLU_Z};

  };

  template<>
  struct GetSuperLUType<std::complex<float> >
  {
    enum { type = SLU_C};

  };


  /**
   * @brief Utility class for converting an ISTL Matrix
   * into a SsuperLU Matrix.
   */
  template<class M>
  struct SuperLUMatrix
  {};

  template<class M>
  struct SuperMatrixInitializer
  {};

  template<class M, class X, class TM, class T1>
  class SeqOverlappingSchwarz;

  /**
   * @brief Coverte for BCRSMatrix to SuperLU Matrix.
   */
  template<class B, class TA, int n, int m>
  class SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  {
    template<class M, class X, class TM, class T1>
    friend class SeqOverlappingSchwarz;
    friend class SuperMatrixInitializer<BCRSMatrix<FieldMatrix<B,n,m>,TA> >;

  public:
    /** @brief The type of the matrix to convert. */
    typedef BCRSMatrix<FieldMatrix<B,n,m>,TA> Matrix;

    typedef typename Matrix::size_type size_type;

    /**
     * @brief Constructor that initializes the data.
     * @param mat The matrix to convert.
     */
    SuperLUMatrix(const Matrix& mat);

    SuperLUMatrix();

    /** @brief Destructor */
    ~SuperLUMatrix();

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
    bool operator==(const Matrix& mat) const;

    /**
     * @brief Get the number of rows.
     * @return  The number of rows.
     */
    size_type N() const
    {
      return N_;
    }

    /**
     * @brief Get the number of columns.
     * @return  The number of columns.
     */
    size_type M() const
    {
      return M_;
    }

    SuperLUMatrix& operator=(const Matrix& mat);

    SuperLUMatrix& operator=(const SuperLUMatrix& mat);

  private:
    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& mat);
    /** @brief free allocated space. */
    void free();

    int N_, M_, Nnz_;
    B* values;
    int* rowindex;
    int* colstart;
    SuperMatrix A;
  };

  template<class T, class A, int n, int m>
  void writeCompColMatrixToMatlab(const SuperLUMatrix<BCRSMatrix<FieldMatrix<T,n,m>,A> >& mat,
                                  std::ostream& os)
  {
    const SuperMatrix a=static_cast<const SuperMatrix&>(mat);
    NCformat *astore = (NCformat *) a.Store;
    double* dp = (double*)astore->nzval;

    // remember old flags
    std::ios_base::fmtflags oldflags = os.flags();
    // set the output format
    //os.setf(std::ios_base::scientific, std::ios_base::floatfield);
    int oldprec = os.precision();
    //os.precision(10);
    //dPrint_CompCol_Matrix("A", const_cast<SuperMatrix*>(&a));

    os <<"[";
    for(int row=0; row<a.nrow; ++row) {
      for(int col= 0; col < a.ncol; ++col) {
        // linear search for col
        int i;
        for(i=astore->colptr[col]; i < astore->colptr[col+1]; ++i)
          if(astore->rowind[i]==row) {
            os<<dp[i]<<" ";
            break;
          }
        if(i==astore->colptr[col+1])
          // entry not present
          os<<0<<" ";
      }
      if(row==a.nrow-1)
        os<<"]";
      os<<std::endl;
    }
    // reset the output format
    os.flags(oldflags);
    os.precision(oldprec);
  }


  template<class T, class A, int n, int m>
  class SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  {
    template<class I, class S>
    friend class OverlappingSchwarzInitializer;
  public:
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;
    typedef typename Matrix::const_iterator Iter;
    typedef typename Matrix::row_type::const_iterator CIter;
    typedef typename Matrix::size_type size_type;

    SuperMatrixInitializer(SuperLUMatrix& m);

    SuperMatrixInitializer();

    ~SuperMatrixInitializer();

    void addRowNnz(const Iter& row) const;

    void allocate();

    void countEntries(const Iter& row, const CIter& col) const;

    void countEntries(size_type colidx) const;

    void calcColstart() const;

    void copyValue(const Iter& row, const CIter& col) const;

    void copyValue(const CIter& col, size_type rowindex, size_type colidx) const;

    void createMatrix() const;

  private:

    void allocateMatrixStorage() const;

    void allocateMarker();

    SuperLUMatrix* mat;
    int cols;
    typename Matrix::size_type *marker;
  };

  template<class T, class A, int n, int m>
  SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::SuperMatrixInitializer(SuperLUMatrix& mat_)
    : mat(&mat_), cols(mat_.M()), marker(0)
  {
    mat->Nnz_=0;
  }

  template<class T, class A, int n, int m>
  SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::SuperMatrixInitializer()
    : mat(0), cols(0), marker(0)
  {}

  template<class T, class A, int n, int m>
  SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::~SuperMatrixInitializer()
  {
    if(marker)
      delete[] marker;
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::addRowNnz(const Iter& row) const
  {
    mat->Nnz_+=row->getsize();
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocate()
  {
    allocateMatrixStorage();
    allocateMarker();
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocateMatrixStorage() const
  {
    mat->Nnz_*=n*m;
    // initialize data
    mat->values=new T[mat->Nnz_];
    mat->rowindex=new int[mat->Nnz_];
    mat->colstart=new int[cols+1];
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::allocateMarker()
  {
    marker = new typename Matrix::size_type[cols];

    for(int i=0; i < cols; ++i)
      marker[i]=0;
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::countEntries(const Iter& row, const CIter& col) const
  {
    countEntries(col.index());

  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::countEntries(size_type colindex) const
  {
    for(int i=0; i < m; ++i)
      marker[colindex*m+i]+=n;
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::calcColstart() const
  {
    mat->colstart[0]=0;
    for(int i=0; i < cols; ++i) {
      mat->colstart[i+1]=mat->colstart[i]+marker[i];
      marker[i]=mat->colstart[i];
    }
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::copyValue(const Iter& row, const CIter& col) const
  {
    copyValue(col, row.index(), col.index());
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::copyValue(const CIter& col, size_type rowindex, size_type colindex) const
  {
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) {
        mat->rowindex[marker[colindex*m+j]]=rowindex*n+i;
        mat->values[marker[colindex*m+j]]=(*col)[i][j];
        ++marker[colindex*m+j];
      }
    }
  }

  template<class T, class A, int n, int m>
  void SuperMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> >::createMatrix() const
  {
    dCreate_CompCol_Matrix(&mat->A, mat->N_, mat->M_, mat->Nnz_,
                           mat->values, mat->rowindex, mat->colstart, SLU_NC, static_cast<Dtype_t>(GetSuperLUType<T>::type), SLU_GE);
  }

  template<class F, class Matrix>
  void copyToSuperMatrix(F& initializer, const Matrix& mat)
  {
    typedef typename Matrix::const_iterator Iter;

    for(Iter row=mat.begin(); row!= mat.end(); ++row)
      initializer.addRowNnz(row);

    initializer.allocate();

    for(Iter row=mat.begin(); row!= mat.end(); ++row) {
      typedef typename  Matrix::row_type::const_iterator CIter;
      for(CIter col=row->begin(); col != row->end(); ++col)
        initializer.countEntries(row, col);
    }

    initializer.calcColstart();

    for(Iter row=mat.begin(); row!= mat.end(); ++row) {
      typedef typename  Matrix::row_type::const_iterator CIter;
      for(CIter col=row->begin(); col != row->end(); ++col)
        initializer.copyValue(row, col);
    }
    initializer.createMatrix();
  }


  template<class B, class TA, int n, int m>
  bool SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator==(const BCRSMatrix<FieldMatrix<B,n,m>,TA>& mat) const
  {
    const NCformat* S=static_cast<const NCformat *>(A.Store);
    for(int col=0; col < M(); ++col) {
      for(int j=S->colptr[col]; j < S->colptr[col+1]; ++j) {
        int row=S->rowind[j];
        if((mat[row/n][col/m])[row%n][col%m]!=reinterpret_cast<B*>(S->nzval)[j]) {
          std::cerr<<" bcrs["<<row/n<<"]["<<col/m<<"]["<<row%n<<"]["<<row%m
                   <<"]="<<(mat[row/n][col/m])[row%n][col%m]<<" super["<<row<<"]["<<col<<"]="<<reinterpret_cast<B*>(S->nzval)[j];

          return false;
        }
      }
    }

    return true;
  }

  template<class B, class TA, int n, int m>
  bool operator==(SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >& sla, BCRSMatrix<FieldMatrix<B,n,m>,TA>& a)
  {
    return a==sla;
  }

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::SuperLUMatrix()
    : N_(0), M_(0), Nnz_(0), values(0), rowindex(0), colstart(0)
  {}

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::SuperLUMatrix(const Matrix& mat)
    : N_(n*mat.N()), M_(m*mat.M()), Nnz_(n*m*mat.Nnz())
  {}

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >&
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(const Matrix& mat)
  {
    if(N_+M_+Nnz_!=0)
      free();
    setMatrix(mat);
    return *this;
  }

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >&
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator=(const SuperLUMatrix& mat)
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
    if(M_+Nnz_>0)
      dCreate_CompCol_Matrix(&A, N_, M_, Nnz_,
                             values, rowindex, colstart, SLU_NC, static_cast<Dtype_t>(GetSuperLUType<B>::type), SLU_GE);
    return *this;
  }

  template<class B, class TA, int n, int m>
  void SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::setMatrix(const Matrix& mat)
  {
    N_=n*mat.N();
    M_=m*mat.M();
    SuperMatrixInitializer<Matrix> initializer(*this);

    copyToSuperMatrix(initializer,mat);

#ifdef DUNE_ISTL_WITH_CHECKING
    char name[] = {'A',0};
    if(N_<0)
      dPrint_CompCol_Matrix(name,&A);
    assert(*this==mat);
#endif
  }

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::~SuperLUMatrix()
  {
    if(N_+M_+Nnz_!=0)
      free();
  }

  template<class B, class TA, int n, int m>
  void SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::free()
  {
    delete[] values;
    delete[] rowindex;
    delete[] colstart;
    SUPERLU_FREE(A.Store);
  }

}
#endif
#endif
