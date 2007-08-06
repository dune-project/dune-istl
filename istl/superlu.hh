// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SUPERLU_HH
#define DUNE_SUPERLU_HH

#ifdef HAVE_SUPERLU

#include "dsp_defs.h"
#include "solvers.hh"
#include "supermatrix.hh"
#include <algorithm>
#include <functional>
#include "bcrsmatrix.hh"
#include "bvector.hh"
#include "istlexception.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/stdstreams.hh>

namespace Dune
{

  template<class Matrix>
  class SuperLU
  {};


  template<typename T, typename A, int n, int m>
  class SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A > >
    : public InverseOperator<BlockVector<FieldVector<T,m>,A>,BlockVector<FieldVector<T,n>,A> >
  {
  public:
    typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef SuperLUMatrix<Matrix> SuperLUMatrix;
    typedef BlockVector<FieldVector<T,m>,A> domain_type;
    typedef BlockVector<FieldVector<T,n>,A> range_type;
    /**
     * @brief Constructor.
     * @param mat The matrix of the system to solve.
     */
    explicit SuperLU(const Matrix& mat);
    /**
     * @brief Empty default constructor.
     *
     * Use setMatrix to tell SuperLU for what matrix it solves.
     */
    SuperLU();

    ~SuperLU();

    /**
     *
     */
    void apply(domain_type& x, range_type& b, InverseOperatorResult& res);

    /**
     *
     */
    void apply (domain_type& x, range_type& b, double reduction, InverseOperatorResult& res)
    {
      apply(x,b,res);
      res.converged=res.reduction<reduction;
    }


    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& mat);

  private:
    /** @brief free allocated space. */
    void free();

    const Matrix* a;
    SuperLUMatrix mat;
    SuperMatrix L, U, B, X;
    int *perm_c, *perm_r, *etree;
    T *R, *C;
    T *bstore;
    superlu_options_t options;
    char equed;
    void *work;
    int lwork;
    bool first;
  };


  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::~SuperLU()
  {
    if(a)
      free();
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::free()
  {
    delete[] perm_c;
    delete[] perm_r;
    delete[] etree;
    delete[] R;
    delete[] C;
    if(lwork>=0) {
      Destroy_SuperNode_Matrix(&L);
      Destroy_CompCol_Matrix(&U);
    }
    lwork=0;
    if(!first) {
      SUPERLU_FREE(B.Store);
      SUPERLU_FREE(X.Store);
    }
    a=0;
  }

  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::SuperLU(const Matrix& mat_)
    : a(0), lwork(0), work(0), first(true)
  {
    setMatrix(mat_);

  }
  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::SuperLU()
    : a(0)
  {}

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::setMatrix(const Matrix& mat_)
  {
    if(a) {
      free();
    }
    lwork=0;
    work=0;
    a=&mat_;
    mat=mat_;
    first = true;
    perm_c = new int[mat.M()];
    perm_r = new int[mat.N()];
    etree  = new int[mat.M()];
    R = new double[mat.N()];
    C = new double[mat.M()];

    set_default_options(&options);
    // Do the factorization
    B.ncol=0;
    B.Stype=SLU_DN;
    B.Dtype=SLU_D;
    B.Mtype= SLU_GE;
    DNformat fakeFormat;
    fakeFormat.lda=mat.N();
    B.Store=&fakeFormat;
    X.Stype=SLU_DN;
    X.Dtype=SLU_D;
    X.Mtype= SLU_GE;
    X.ncol=0;
    X.Store=&fakeFormat;

    double rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;

    StatInit(&stat);
    dgssvx(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr,
           &berr, &memusage, &stat, &info);

    dinfo<<"LU factorization: dgssvx() returns info "<< info<<std::endl;

    if ( info == 0 || info == n+1 ) {

      if ( options.PivotGrowth )
        dinfo<<"Recip. pivot growth = "<<rpg<<std::endl;
      if ( options.ConditionNumber )
        dinfo<<"Recip. condition number = %e\n"<< rcond<<std::endl;
      SCformat* Lstore = (SCformat *) L.Store;
      NCformat* Ustore = (NCformat *) U.Store;
      dinfo<<"No of nonzeros in matrix ="<<mat.Nnz()<<std::endl;
      dinfo<<"No of nonzeros in factor L = "<< Lstore->nnz<<std::endl;
      dinfo<<"No of nonzeros in factor U = "<< Ustore->nnz<<std::endl;
      dinfo<<"No of nonzeros in L+U = "<< Lstore->nnz + Ustore->nnz - n<<std::endl;
      dinfo<<"L\\U MB "<<memusage.for_lu/1e6<<" \ttotal MB needed "<<memusage.total_needed/1e6
           <<" \texpansions "<<memusage.expansions<<std::endl;
    } else if ( info > 0 && lwork == -1 ) {
      dinfo<<"** Estimated memory: "<< info - n<<std::endl;
    }
    if ( options.PrintStat ) StatPrint(&stat);
    StatFree(&stat);
    options.Fact = FACTORED;
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(domain_type& x, range_type& b, InverseOperatorResult& res)
  {
    if(!a)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    if(first) {
      dCreate_Dense_Matrix(&B, mat.N(), 1,  reinterpret_cast<T*>(&b[0]), mat.N(), SLU_DN, SLU_D, SLU_GE);
      dCreate_Dense_Matrix(&X, mat.N(), 1,  reinterpret_cast<T*>(&x[0]), mat.N(), SLU_DN, SLU_D, SLU_GE);
      first=false;
    }else{
      ((DNformat*) B.Store)->nzval=&b[0];
      ((DNformat*)X.Store)->nzval=&x[0];
    }

    double rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;
    /* Initialize the statistics variables. */
    StatInit(&stat);

    range_type d=b;
    a->usmv(-1, x, d);

    double def0=d.two_norm();

    dgssvx(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr, &berr,
           &memusage, &stat, &info);

    dinfo<<"Triangular solve: dgssvx() returns info "<< info<<std::endl;


    res.iterations=1;

    if(options.Equil==YES)
      // undo scaling of right hand side
      std::transform(reinterpret_cast<T*>(&b[0]),reinterpret_cast<T*>(&b[0])+mat.M(),
                     &C[0], reinterpret_cast<T*>(&d[0]), std::divides<T>());
    else
      d=b;

    a->usmv(-1, x, d);
    res.reduction=d.two_norm()/def0;
    res.conv_rate = res.reduction;

    if ( info == 0 || info == n+1 ) {

      if ( options.IterRefine ) {
        dinfo<<"Iterative Refinement: steps="
             <<stat.RefineSteps<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
      }
    } else if ( info > 0 && lwork == -1 ) {
      dinfo<<"** Estimated memory: "<< info - n<<" bytes"<<std::endl;
    }
    if ( options.PrintStat ) StatPrint(&stat);
    StatFree(&stat);
  }

};

#endif
#endif
