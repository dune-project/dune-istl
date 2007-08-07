// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SUPERLUMATRIX_HH
#include "dsp_defs.h"
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

  /**
   * @brief Coverte for BCRSMatrix to SuperLU Matrix.
   */
  template<class B, class TA, int n, int m>
  class SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  {
  public:
    /** @brief The type of the matrix to convert. */
    typedef BCRSMatrix<FieldMatrix<B,n,m>,TA> Matrix;

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

    bool operator==(const Matrix& mat) const;

    /**
     * @brief Get the number of rows.
     * @return  The number of rows.
     */
    std::size_t N() const
    {
      return N_;
    }

    /**
     * @brief Get the number of columns.
     * @return  The number of columns.
     */
    std::size_t M() const
    {
      return M_;
    }

    std::size_t Nnz() const
    {
      return Nnz_;
    }

    SuperLUMatrix& operator=(const Matrix& mat);

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

  /*
     template<class B, class TA, int n>
     class SuperLUMatrix<BlockVector<FieldVector<B,n>,TA> >
     {
     public:
     typedef BlockVector<FieldVector<B,n>,TA> Vector;

     SuperLUMatrix(const Vector& v)
     {
      // allocate storage
      a=new double[v.size()*n];
      dCreate_Dense_Matrix(&b, v.size()*n, 1, v.size()*n, SLU_DN, GetSuperLUType<B>::type, SLU_GE);
     }
     ~SuperLUMatrix()
     {
      Destroy_SuperMatrix_Store(&b);
      delete[] a;
     }

     private:
     double* a;
     SuperMatrix b;
     };

   */

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
  void SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::setMatrix(const Matrix& mat)
  {
    N_=n*mat.N();
    M_=m*mat.M();
    // Calculate no of nonzeros
    Nnz_=0;
    typedef typename Matrix::ConstRowIterator Iter;

    for(Iter row=mat.begin(); row!= mat.end(); ++row)
      Nnz_+=row->getsize();
    Nnz_*=n*m;

    // initialize data
    values=new B[Nnz_];
    rowindex=new int[Nnz_];
    colstart=new int[mat.M()*m+1];

    // calculate pattern for the transposed matrix
    typedef typename Matrix::row_type row_type;

    std::size_t* marker = new std::size_t[mat.M()*m];

    for(int i=0; i < m*mat.M(); ++i)
      marker[i]=0;

    for(Iter row=mat.begin(); row!= mat.end(); ++row) {
      typedef typename row_type::const_iterator CIter;
      for(CIter col=row->begin(); col != row->end(); ++col) {
        for(int i=0; i < m; ++i)
          marker[col.index()*m+i]+=n;
      }
    }

    // convert no rownnz to colstart
    colstart[0]=0;
    for(int i=0; i < m*mat.M(); ++i) {
      colstart[i+1]=colstart[i]+marker[i];
      marker[i]=colstart[i];
    }

    // copy data
    for(Iter row=mat.begin(); row!= mat.end(); ++row) {
      typedef typename row_type::const_iterator CIter;
      for(int i=0; i<n; i++) {
        for(CIter col=row->begin(); col != row->end(); ++col) {
          for(int j=0; j<m; j++) {
            //std::cout<<col.index()<<"*"<<m<<"+"<<j<<"="<<col.index()*m+j<<" ";
            //std::cout<<"marker="<<marker[col.index()*m+j]<<std::endl;
            //std::cout<<"rowindex="<<rowindex[0]<<std::endl;
            //std::cout<<rowindex[marker[col.index()*m+j]]<<std::endl;
            //std::cout<<row.index()<<std::endl;
            //std::cout<<col.index()<<std::endl;
            //std::cout<<col.index()*m+j<<std::endl;
            //std::cout<<marker[col.index()*m+j]<<std::endl;
            rowindex[marker[col.index()*m+j]]=row.index()*n+i;
            values[marker[col.index()*m+j]]=(*col)[i][j];
            ++marker[col.index()*m+j];
          }
        }
      }
    }
    delete[] marker;

    dCreate_CompCol_Matrix(&A, N_, M_, Nnz_,
                           values, rowindex, colstart, SLU_NC, static_cast<Dtype_t>(GetSuperLUType<B>::type), SLU_GE);
#ifdef DUNE_ISTL_WITH_CHECKING
    if(N_<30)
      dPrint_CompCol_Matrix("A",&A);
    assert(*this==mat);
#endif
  }

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::~SuperLUMatrix()
  {
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

};

#endif
