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

    /** @brief Destructor */
    ~SuperLUMatrix();

    /** @brief Cast to a SuperLU Matrix */
    operator SuperMatrix&()
    {
      return A;
    }

    bool operator==(const BCRSMatrix<FieldMatrix<B,n,m>,TA>& mat);

    /**
     * @brief Get the number of rows.
     * @return  The number of rows.
     */
    std::size_t N()
    {
      return N_;
    }

    /**
     * @brief Get the number of columns.
     * @return  The number of columns.
     */
    std::size_t M()
    {
      return M_;
    }

  private:
    int N_, M_;
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
  bool SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::operator==(const BCRSMatrix<FieldMatrix<B,n,m>,TA>& mat)
  {
    const NCformat* S=static_cast<const NCformat *>(A.Store);
    for(int col=0; col < M(); ++col) {
      for(int j=S->colptr[col]; j < S->colptr[col+1]; ++j) {
        int row=S->rowind[j];
        if((mat[row/n][col/m])[row%n][row%m]!=reinterpret_cast<B*>(S->nzval)[j])
          return false;
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
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >
  ::SuperLUMatrix(const Matrix& mat)
    : N_(n*mat.N()), M_(m*mat.M())
  {
    // initialize data
    values=new B[mat.Nnz()*n*m];
    rowindex=new int[mat.Nnz()*n*m];
    colstart=new int[mat.M()*m+1];

    // calculate pattern for the transposed matrix
    typedef typename Matrix::ConstRowIterator Iter;
    typedef typename Matrix::row_type row_type;

    std::size_t* marker = new std::size_t[mat.M()*m];

    for(int i=0; i < m*mat.M(); ++i)
      marker[i]=0;

    for(Iter row=mat.begin(); row!= mat.end(); ++row) {
      typedef typename row_type::const_iterator CIter;
      for(CIter col=row->begin(); col != row->end(); ++col) {
        for(int i=0; i < m; ++i)
          ++marker[col.index()*m+i];
      }
    }

    // convert no rownnz to colstart
    colstart[0]=0;
    for(int i=0; i < m*mat.M(); ++i) {
      for(int j=0; j<m; j++) {
        colstart[i*m+j+1]=colstart[i*m+j]+marker[i*m+j];
        marker[i*m+j]=colstart[i*m+j];
      }
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

            rowindex[marker[col.index()*m+j]]=row.index()*n+i;
            values[marker[col.index()*m+j]]=(*col)[i][j];
            ++marker[col.index()*m+j];
          }
        }
      }
    }
    delete[] marker;

    dCreate_CompCol_Matrix(&A, mat.N(), mat.M(), mat.Nnz(),
                           values, rowindex, colstart, SLU_NC, static_cast<Dtype_t>(GetSuperLUType<B>::type), SLU_GE);
    assert(*this==mat);
    dPrint_CompCol_Matrix("A", &A);
  }

  template<class B, class TA, int n, int m>
  SuperLUMatrix<BCRSMatrix<FieldMatrix<B,n,m>,TA> >::~SuperLUMatrix()
  {
    delete[] values;
    delete[] rowindex;
    delete[] colstart;
    SUPERLU_FREE(A.Store);
  }

};

#endif
