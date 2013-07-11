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
#ifndef SUPERLU_NTYPE
#define  SUPERLU_NTYPE 1
#endif
#ifdef SUPERLU_POST_2005_VERSION

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

#else

#if SUPERLU_NTYPE==0
#include "ssp_defs.h"
#endif

#if SUPERLU_NTYPE==1
#include "dsp_defs.h"
#warning Support for SuperLU older than SuperLU 3.0 from August 2005 is deprecated.
#endif

#if SUPERLU_NTYPE==2
#include "csp_defs.h"
#endif

#if SUPERLU_NTYPE>=3
#include "zsp_defs.h"
#endif

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
#include <dune/istl/solvertype.hh>

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
  struct SeqOverlappingSchwarzAssembler;

  template<class T>
  struct SuperLUSolveChooser
  {};

  template<class T>
  struct SuperLUDenseMatChooser
  {};

  template<class T>
  struct SuperLUQueryChooser
  {};

  template<class T>
  struct QuerySpaceChooser
  {};

#if SUPERLU_NTYPE==0
  template<>
  struct SuperLUDenseMatChooser<float>
  {
    static void create(SuperMatrix *mat, int n, int m, float *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      sCreate_Dense_Matrix(mat, n, m, dat, n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix *m)
    {}

  };
  template<>
  struct SuperLUSolveChooser<float>
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      float *rpg, float *rcond, float *ferr, float *berr,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
      sgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
    }
  };

  template<>
  struct QuerySpaceChooser<float>
  {
    static void querySpace(SuperMatrix* L, SuperMatrix* U, mem_usage_t* memusage)
    {
      sQuerySpace(L,U,memusage);
    }
  };

#endif

#if SUPERLU_NTYPE==1

  template<>
  struct SuperLUDenseMatChooser<double>
  {
    static void create(SuperMatrix *mat, int n, int m, double *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      dCreate_Dense_Matrix(mat, n, m, dat, n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix *mat)
    {}
  };
  template<>
  struct SuperLUSolveChooser<double>
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, double *R, double *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      double *rpg, double *rcond, double *ferr, double *berr,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
      dgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
    }
  };

  template<>
  struct QuerySpaceChooser<double>
  {
    static void querySpace(SuperMatrix* L, SuperMatrix* U, mem_usage_t* memusage)
    {
      dQuerySpace(L,U,memusage);
    }
  };
#endif

#if SUPERLU_NTYPE>=3
  template<>
  struct SuperLUDenseMatChooser<std::complex<double> >
  {
    static void create(SuperMatrix *mat, int n, int m, std::complex<double> *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      zCreate_Dense_Matrix(mat, n, m, reinterpret_cast<doublecomplex*>(dat), n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix *mat)
    {}
  };

  template<>
  struct SuperLUSolveChooser<std::complex<double> >
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, double *R, double *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      double *rpg, double *rcond, double *ferr, double *berr,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
      zgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
    }
  };

  template<>
  struct QuerySpaceChooser<std::complex<double> >
  {
    static void querySpace(SuperMatrix* L, SuperMatrix* U, mem_usage_t* memusage)
    {
      zQuerySpace(L,U,memusage);
    }
  };
#endif

#if SUPERLU_NTYPE==2
  template<>
  struct SuperLUDenseMatChooser<std::complex<float> >
  {
    static void create(SuperMatrix *mat, int n, int m, std::complex<float> *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      cCreate_Dense_Matrix(mat, n, m, reinterpret_cast< ::complex*>(dat), n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix *mat)
    {}
  };

  template<>
  struct SuperLUSolveChooser<std::complex<float> >
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      float *rpg, float *rcond, float *ferr, float *berr,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
      cgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
    }
  };

  template<>
  struct QuerySpaceChooser<std::complex<float> >
  {
    static void querySpace(SuperMatrix* L, SuperMatrix* U, mem_usage_t* memusage)
    {
      cQuerySpace(L,U,memusage);
    }
  };
#endif

  /**
   * @brief SuperLu Solver
   *
   * Uses the well known <a href="http://crd.lbl.gov/~xiaoye/SuperLU/">SuperLU
   * package</a> to solve the
   * system.
   *
   * SuperLU supports single and double precision floating point and complex
   * numbers. Unfortunately these cannot be used at the same time.
   * Therfore users must set SUPERLU_NTYPE (0: float, 1: double,
   * 2: std::complex<float>, 3: std::complex<double>)
   * if the numeric type should be different from double.
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
     * @param reusevector Default value is true. If true the two vectors are allocate in
     * the first call to apply. These get resused in subsequent calls to apply
     * and are deallocated in the destructor. If false these vectors are allocated
     * at the beginning and deallocated at the end of each apply method. This allows
     * using the same instance of superlu from different threads.
     */
    explicit SuperLU(const Matrix& mat, bool verbose=false,
                     bool reusevector=true);
    /**
     * @brief Empty default constructor.
     *
     * Use setMatrix to tell SuperLU for what matrix it solves.
     * Using this constructor no vectors will be reused.
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
    friend struct SeqOverlappingSchwarzAssembler<SuperLU<Matrix> >;

    /** @brief computes the LU Decomposition */
    void decompose();

    SuperLUMatrix mat;
    SuperMatrix L, U, B, X;
    int *perm_c, *perm_r, *etree;
    typename GetSuperLUType<T>::float_type *R, *C;
    T *bstore;
    superlu_options_t options;
    char equed;
    void *work;
    int lwork;
    bool first, verbose, reusevector;
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
    if(!first && reusevector) {
      SUPERLU_FREE(B.Store);
      SUPERLU_FREE(X.Store);
    }
    mat.free();
  }

  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::SuperLU(const Matrix& mat_, bool verbose_, bool reusevector_)
    : work(0), lwork(0), first(true), verbose(verbose_),
      reusevector(reusevector_)
  {
    setMatrix(mat_);

  }
  template<typename T, typename A, int n, int m>
  SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::SuperLU()
    :    work(0), lwork(0),verbose(false),
      reusevector(false)
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
    R = new typename GetSuperLUType<T>::float_type[mat.N()];
    C = new typename GetSuperLUType<T>::float_type[mat.M()];

    set_default_options(&options);
    // Do the factorization
    B.ncol=0;
    B.Stype=SLU_DN;
    B.Dtype=GetSuperLUType<T>::type;
    B.Mtype= SLU_GE;
    DNformat fakeFormat;
    fakeFormat.lda=mat.N();
    B.Store=&fakeFormat;
    X.Stype=SLU_DN;
    X.Dtype=GetSuperLUType<T>::type;
    X.Mtype= SLU_GE;
    X.ncol=0;
    X.Store=&fakeFormat;

    typename GetSuperLUType<T>::float_type rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;

    StatInit(&stat);
    SuperLUSolveChooser<T>::solve(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
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
        QuerySpaceChooser<T>::querySpace(&L, &U, &memusage);
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

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(domain_type& x, range_type& b, InverseOperatorResult& res)
  {
    if(mat.M()+mat.N()==0)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    SuperMatrix* mB = &B;
    SuperMatrix* mX = &X;
    SuperMatrix rB, rX;
    if (reusevector) {
      if(first) {
        SuperLUDenseMatChooser<T>::create(&B, (int)mat.N(), 1,  reinterpret_cast<T*>(&b[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
        SuperLUDenseMatChooser<T>::create(&X, (int)mat.N(), 1,  reinterpret_cast<T*>(&x[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
        first=false;
      }else{
        ((DNformat*) B.Store)->nzval=&b[0];
        ((DNformat*)X.Store)->nzval=&x[0];
      }
    } else {
      SuperLUDenseMatChooser<T>::create(&rB, (int)mat.N(), 1,  reinterpret_cast<T*>(&b[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      SuperLUDenseMatChooser<T>::create(&rX, (int)mat.N(), 1,  reinterpret_cast<T*>(&x[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      mB = &rB;
      mX = &rX;
    }
    typename GetSuperLUType<T>::float_type rpg, rcond, ferr, berr;
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
#ifdef SUPERLU_MIN_VERSION_4_3
    options.IterRefine=SLU_DOUBLE;
#else
    options.IterRefine=DOUBLE;
#endif

    SuperLUSolveChooser<T>::solve(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
                                  &L, &U, work, lwork, mB, mX, &rpg, &rcond, &ferr, &berr,
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
    if (!reusevector) {
      SUPERLU_FREE(rB.Store);
      SUPERLU_FREE(rX.Store);
    }
  }

  template<typename T, typename A, int n, int m>
  void SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >
  ::apply(T* x, T* b)
  {
    if(mat.N()+mat.M()==0)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    SuperMatrix& mB = B;
    SuperMatrix& mX = X;
    SuperMatrix rB, rX;
    if (reusevector) {
      if(first) {
        SuperLUDenseMatChooser<T>::create(&B, mat.N(), 1,  b, mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
        SuperLUDenseMatChooser<T>::create(&X, mat.N(), 1,  x, mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
        first=false;
      }else{
        ((DNformat*) B.Store)->nzval=b;
        ((DNformat*)X.Store)->nzval=x;
      }
    } else {
      SuperLUDenseMatChooser<T>::create(&rB, mat.N(), 1,  b, mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      SuperLUDenseMatChooser<T>::create(&rX, mat.N(), 1,  x, mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      mB = rB;
      mX = rX;
    }

    typename GetSuperLUType<T>::float_type rpg, rcond, ferr, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;
    /* Initialize the statistics variables. */
    StatInit(&stat);

#ifdef SUPERLU_MIN_VERSION_4_3
    options.IterRefine=SLU_DOUBLE;
#else
    options.IterRefine=DOUBLE;
#endif

    SuperLUSolveChooser<T>::solve(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
                                  &L, &U, work, lwork, &mB, &mX, &rpg, &rcond, &ferr, &berr,
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
    if (!reusevector) {
      SUPERLU_FREE(rB.Store);
      SUPERLU_FREE(rX.Store);
    }
  }
  /** @} */

  template<typename T, typename A, int n, int m>
  struct IsDirectSolver<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >
  {
    enum { value=true};
  };
}

#endif // HAVE_SUPERLU
#endif // DUNE_SUPERLU_HH
