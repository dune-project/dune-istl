// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_SUPERLU_HH
#define DUNE_SUPERLU_HH

#if HAVE_SUPERLU
#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif
#ifdef SUPERLU_POST_2005_VERSION
#include "slu_ddefs.h"
//#include "slu_sdefs.h"
#else
#include "dsp_defs.h"
//#include "fsp_defs.h"
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

  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T>
  class SeqOverlappingSchwarzAssembler;

  /**
   * @brief SuperLu Solver
   *
   * Uses the well known <a href="http://crd.lbl.gov/~xiaoye/SuperLU/">SuperLU
   * package</a> to solve the
   * system.
   */
  template<typename T, typename A, int n, int m>
  class SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A > >
    : public InverseOperator<
          BlockVector<FieldVector<T,m>,
              typename A::template rebind<FieldVector<T,m> >::other>,
          BlockVector<FieldVector<T,n>,
              typename A::template rebind<FieldVector<T,n> >::other> >
  {
  public:
    /* @brief The matrix type. */
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    /* @brief The corresponding SuperLU Matrix type.*/
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;
    /** @brief The type of the domain of the solver. */
    typedef Dune::BlockVector<
        FieldVector<T,m>,
        typename A::template rebind<FieldVector<T,m> >::other> domain_type;
    /** @brief The type of the range of the solver. */
    typedef Dune::BlockVector<
        FieldVector<T,n>,
        typename A::template rebind<FieldVector<T,n> >::other> range_type;
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
    void apply (domain_type& x, range_type& b, typename Dune::FieldTraits<T>::real_type reduction,
                InverseOperatorResult& res)
    {
      apply(x,b,res);
    }

    /**
     * @brief Apply SuperLu to C arrays.
     */
    void apply(T* x, T* b);

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& mat);

    typename SuperLUMatrix::size_type nnz() const
    {
      return mat.nnz();
    }

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
    template<class M,class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;
    friend class SeqOverlappingSchwarzAssembler<SuperLU<Matrix> >;

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
    R = new T[mat.N()];
    C = new T[mat.M()];

    set_default_options(&options);
    // Do the factorization
    B.ncol=0;
    B.Stype=SLU_DN;
    B.Dtype= static_cast<Dtype_t>(GetSuperLUType<T>::type);
    B.Mtype= SLU_GE;
    DNformat fakeFormat;
    fakeFormat.lda=mat.N();
    B.Store=&fakeFormat;
    X.Stype=SLU_DN;
    X.Dtype=static_cast<Dtype_t>(GetSuperLUType<T>::type);
    X.Mtype= SLU_GE;
    X.ncol=0;
    X.Store=&fakeFormat;

    T rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;

    StatInit(&stat);
    applySuperLU(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
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
        dQuerySpace(&L, &U, &memusage);
        dinfo<<"L\\U MB "<<memusage.for_lu/1e6<<" \ttotal MB needed "<<memusage.total_needed/1e6
             <<" \texpansions ";

#ifdef HAVE_MEM_USAGE_T_EXPANSIONS
        std::cout<<memusage.expansions<<std::endl;
#else
        std::cout<<stat.expansions<<std::endl;
#endif
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

  void createDenseSuperLUMatrix(SuperMatrix* B, int rows, int cols, double* b, int size,
                                Stype_t stype, Mtype_t mtype)
  {
    dCreate_Dense_Matrix(B, rows, cols, b, size, stype, SLU_D, mtype);
  }

  // Unfortunately SuperLU uses a lot of copy and paste in its headers.
  // This results in some structs being declares in the headers of the float
  // AND double version. To get around this we only include the double version
  // and define the functions of the other versions as extern.
  extern "C"
  {
    // single precision versions of SuperLU
    void sCreate_Dense_Matrix(SuperMatrix* B, int rows, int cols, float* b, int size,
                              Stype_t stype, Dtype_t dtype, Mtype_t mtype);


    void sgssvx(superlu_options_t *options, SuperMatrix *mat, int *permc, int *permr, int *etree,
                char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                float *rpg, float *rcond, float *ferr, float *berr,
                mem_usage_t *memusage, SuperLUStat_t *stat, int *info);
  }

  void createDenseSuperLUMatrix(SuperMatrix* B, int rows, int cols, float* b, int size,
                                Stype_t stype, Mtype_t mtype)
  {
    sCreate_Dense_Matrix(B, rows, cols, b, size, stype, SLU_S, mtype);
  }

  void applySuperLU(superlu_options_t *options, SuperMatrix *mat, int *permc, int *permr, int *etree,
                    char *equed, double *R, double *C, SuperMatrix *L, SuperMatrix *U,
                    void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                    double *rpg, double *rcond, double *ferr, double *berr,
                    mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
  {
    dgssvx(options, mat, permc, permr, etree, equed, R, C,
           L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
           memusage, stat, info);
  }


  void applySuperLU(superlu_options_t *options, SuperMatrix *mat, int *permc, int *permr, int *etree,
                    char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                    void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                    float *rpg, float *rcond, float *ferr, float *berr,
                    mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
  {
    sgssvx(options, mat, permc, permr, etree, equed, R, C,
           L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
           memusage, stat, info);
  }
  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(domain_type& x, range_type& b, InverseOperatorResult& res)
  {
    if(mat.M()+mat.N()==0)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    if(first) {
      assert(mat.N()<=static_cast<std::size_t>(std::numeric_limits<int>::max()));
      createDenseSuperLUMatrix(&B, mat.N(), 1,  reinterpret_cast<T*>(&b[0]),
                               mat.N(), SLU_DN, SLU_GE);
      createDenseSuperLUMatrix(&X, mat.N(), 1,  reinterpret_cast<T*>(&x[0]),
                               mat.N(), SLU_DN, SLU_GE);
      first=false;
    }else{
      ((DNformat*) B.Store)->nzval=&b[0];
      ((DNformat*)X.Store)->nzval=&x[0];
    }

    T rpg, rcond, ferr, berr;
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

    applySuperLU(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
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
      createDenseSuperLUMatrix(&B, mat.N(), 1,  b, mat.N(), SLU_DN, SLU_GE);
      createDenseSuperLUMatrix(&X, mat.N(), 1,  x, mat.N(), SLU_DN, SLU_GE);
      first=false;
    }else{
      ((DNformat*) B.Store)->nzval=b;
      ((DNformat*)X.Store)->nzval=x;
    }

    T rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;
    /* Initialize the statistics variables. */
    StatInit(&stat);

    options.IterRefine=DOUBLE;

    applySuperLU(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
                 &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr, &berr,
                 &memusage, &stat, &info);

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
