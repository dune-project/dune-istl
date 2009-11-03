// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SUPERLU_HH
#define DUNE_SUPERLU_HH

#ifdef HAVE_SUPERLU
#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif
#ifdef SUPERLU_POST_2005_VERSION
#include "slu_ddefs.h"
#else
#include "dsp_defs.h"
#endif
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

  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Markus Blatt
   * @brief Classes for using SuperLU with ISTL matrices.
   */
  template<class Matrix>
  class SuperLU
  {};

  template<class M, class T, class TM, bool b, class TA>
  class SeqOverlappingSchwarz;

  /**
   * @brief SuperLu Solver
   *
   * Uses the well known <a href="http://crd.lbl.gov/~xiaoye/SuperLU/">SuperLU
   * package</a> to solve the
   * system.
   */
  template<typename T, typename A, int n, int m>
  class SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A > >
    : public InverseOperator<BlockVector<FieldVector<T,m>,A>,BlockVector<FieldVector<T,n>,A> >
  {
  public:
    /* @brief The matrix type. */
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    /* @brief The corresponding SuperLU Matrix type.*/
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;
    /** @brief The type of the domain of the solver. */
    typedef Dune::BlockVector<FieldVector<T,m>,A> domain_type;
    /** @brief The type of the range of the solver. */
    typedef Dune::BlockVector<FieldVector<T,n>,A> range_type;
    /**
     * @brief Constructs the SuperLU solver.
     *
     * During the construction the matrix will be decomposed.
     * That means that in each apply call forward and backward
     * substitutions take place (and no decomposition).
     * @param mat The matrix of the system to solve.
     * @param verbose If true some statistics are printed.
     */
    explicit SuperLU(const Matrix& mat, bool verbose=false);
    /**
     * @brief Empty default constructor.
     *
     * Use setMatrix to tell SuperLU for what matrix it solves.
     */
    SuperLU();

    ~SuperLU();

    /**
     *  \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    void apply(domain_type& x, range_type& b, InverseOperatorResult& res);

    /**
     *  \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
     */
    void apply (domain_type& x, range_type& b, double reduction, InverseOperatorResult& res)
    {
      apply(x,b,res);
      res.converged=res.reduction<reduction;
    }

    /**
     * @brief Apply SuperLu to C arrays.
     */
    void apply(T* x, T* b);

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& mat);

    template<class S>
    void setSubMatrix(const Matrix& mat, const S& rowIndexSet);

    void setVerbosity(bool v);

    /**
     * @brief free allocated space.
     * @warning later calling apply will result in an error.
     */
    void free();
  private:
    friend class std::mem_fun_ref_t<void,SuperLU>;
    template<class M,class X, class TM, bool b, class T1>
    friend class SeqOverlappingSchwarz;

    /** @brief computes the LU Decomposition */
    void decompose();

    SuperLUMatrix mat;
    SuperMatrix L, U, B, X;
    int *perm_c, *perm_r, *etree;
    T *R, *C;
    T *bstore;
    superlu_options_t options;
    char equed;
    void *work;
    int lwork;
    bool first, verbose;
  };


  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::~SuperLU()
  {
    if(mat.N()+mat.M()>0)
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
    mat.free();
  }

  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::SuperLU(const Matrix& mat_, bool verbose_)
    : work(0), lwork(0), first(true), verbose(verbose_)
  {
    setMatrix(mat_);

  }
  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::SuperLU()
    :    work(0), lwork(0),verbose(false)
  {}
  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::setVerbosity(bool v)
  {
    verbose=v;
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::setMatrix(const Matrix& mat_)
  {
    if(mat.N()+mat.M()>0) {
      free();
    }
    lwork=0;
    work=0;
    //a=&mat_;
    mat=mat_;
    decompose();
  }

  template<typename T, typename A, int n, int m>
  template<class S>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::setSubMatrix(const Matrix& mat_,
                                                                const S& mrs)
  {
    if(mat.N()+mat.M()>0) {
      free();
    }
    lwork=0;
    work=0;
    //a=&mat_;
    mat.setMatrix(mat_,mrs);
    decompose();
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::decompose()
  {

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

    if(verbose) {
      dinfo<<"LU factorization: dgssvx() returns info "<< info<<std::endl;

      if ( info == 0 || info == n+1 ) {

        if ( options.PivotGrowth )
          dinfo<<"Recip. pivot growth = "<<rpg<<std::endl;
        if ( options.ConditionNumber )
          dinfo<<"Recip. condition number = %e\n"<< rcond<<std::endl;
        SCformat* Lstore = (SCformat *) L.Store;
        NCformat* Ustore = (NCformat *) U.Store;
        dinfo<<"No of nonzeros in factor L = "<< Lstore->nnz<<std::endl;
        dinfo<<"No of nonzeros in factor U = "<< Ustore->nnz<<std::endl;
        dinfo<<"No of nonzeros in L+U = "<< Lstore->nnz + Ustore->nnz - n<<std::endl;
        dinfo<<"L\\U MB "<<memusage.for_lu/1e6<<" \ttotal MB needed "<<memusage.total_needed/1e6
             <<" \texpansions "<<memusage.expansions<<std::endl;
      } else if ( info > 0 && lwork == -1 ) {
        dinfo<<"** Estimated memory: "<< info - n<<std::endl;
      }
      if ( options.PrintStat ) StatPrint(&stat);
    }
    StatFree(&stat);
    /*
       NCformat* Ustore = (NCformat *) U.Store;
       int k=0;
       dPrint_CompCol_Matrix("U", &U);
       for(int i=0; i < U.ncol; ++i, ++k){
       std::cout<<i<<": ";
       for(int c=Ustore->colptr[i]; c < Ustore->colptr[i+1]; ++c)
        //if(Ustore->rowind[c]==i)
        std::cout<<Ustore->rowind[c]<<"->"<<((double*)Ustore->nzval)[c]<<" ";
       if(k==0){
        //
        k=-1;
       }std::cout<<std::endl;
       }
       dPrint_SuperNode_Matrix("L", &L);
       for(int i=0; i < U.ncol; ++i, ++k){
       std::cout<<i<<": ";
       for(int c=Ustore->colptr[i]; c < Ustore->colptr[i+1]; ++c)
        //if(Ustore->rowind[c]==i)
        std::cout<<Ustore->rowind[c]<<"->"<<((double*)Ustore->nzval)[c]<<" ";
       if(k==0){
        //
        k=-1;
       }std::cout<<std::endl;
       } */
    options.Fact = FACTORED;
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(domain_type& x, range_type& b, InverseOperatorResult& res)
  {
    if(mat.M()+mat.N()==0)
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
    /*
       range_type d=b;
       a->usmv(-1, x, d);

       double def0=d.two_norm();
     */
    options.IterRefine=DOUBLE;

    dgssvx(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr, &berr,
           &memusage, &stat, &info);

    res.iterations=1;

    /*
       if(options.Equil==YES)
       // undo scaling of right hand side
        std::transform(reinterpret_cast<T*>(&b[0]),reinterpret_cast<T*>(&b[0])+mat.M(),
                       C, reinterpret_cast<T*>(&d[0]), std::divides<T>());
       else
       d=b;
       a->usmv(-1, x, d);
       res.reduction=d.two_norm()/def0;
       res.conv_rate = res.reduction;
       res.converged=(res.reduction<1e-10||d.two_norm()<1e-18);
     */
    res.converged=true;

    if(verbose) {

      dinfo<<"Triangular solve: dgssvx() returns info "<< info<<std::endl;

      if ( info == 0 || info == n+1 ) {

        if ( options.IterRefine ) {
          std::cout<<"Iterative Refinement: steps="
                   <<stat.RefineSteps<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
        }else
          std::cout<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
      } else if ( info > 0 && lwork == -1 ) {
        std::cout<<"** Estimated memory: "<< info - n<<" bytes"<<std::endl;
      }

      if ( options.PrintStat ) StatPrint(&stat);
    }
    StatFree(&stat);
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(T* x, T* b)
  {
    if(mat.N()+mat.M()==0)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    if(first) {
      dCreate_Dense_Matrix(&B, mat.N(), 1,  b, mat.N(), SLU_DN, SLU_D, SLU_GE);
      dCreate_Dense_Matrix(&X, mat.N(), 1,  x, mat.N(), SLU_DN, SLU_D, SLU_GE);
      first=false;
    }else{
      ((DNformat*) B.Store)->nzval=b;
      ((DNformat*)X.Store)->nzval=x;
    }

    double rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;
    /* Initialize the statistics variables. */
    StatInit(&stat);

    options.IterRefine=DOUBLE;

    dgssvx(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr, &berr,
           &memusage, &stat, &info);

    if(options.Equil==YES)
      // undo scaling of right hand side
      std::transform(b, b+mat.M(), C, b, std::divides<T>());

    if(verbose) {
      dinfo<<"Triangular solve: dgssvx() returns info "<< info<<std::endl;

      if ( info == 0 || info == n+1 ) {

        if ( options.IterRefine ) {
          dinfo<<"Iterative Refinement: steps="
               <<stat.RefineSteps<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
        }else
          dinfo<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
      } else if ( info > 0 && lwork == -1 ) {
        dinfo<<"** Estimated memory: "<< info - n<<" bytes"<<std::endl;
      }
      if ( options.PrintStat ) StatPrint(&stat);
    }

    StatFree(&stat);
  }
  /** @} */
}

#endif
#endif
