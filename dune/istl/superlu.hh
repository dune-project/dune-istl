// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SUPERLU_HH
#define DUNE_ISTL_SUPERLU_HH

#if HAVE_SUPERLU

#include "superlufunctions.hh"
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
#include <dune/istl/solverfactory.hh>

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
  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper;

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

#if __has_include("slu_sdefs.h")
  template<>
  struct SuperLUDenseMatChooser<float>
  {
    static void create(SuperMatrix *mat, int n, int m, float *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      sCreate_Dense_Matrix(mat, n, m, dat, n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix*)
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
      GlobalLU_t gLU;
      sgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
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

#if __has_include("slu_ddefs.h")

  template<>
  struct SuperLUDenseMatChooser<double>
  {
    static void create(SuperMatrix *mat, int n, int m, double *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      dCreate_Dense_Matrix(mat, n, m, dat, n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix * /* mat */)
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
      GlobalLU_t gLU;
      dgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
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

#if __has_include("slu_zdefs.h")
  template<>
  struct SuperLUDenseMatChooser<std::complex<double> >
  {
    static void create(SuperMatrix *mat, int n, int m, std::complex<double> *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      zCreate_Dense_Matrix(mat, n, m, reinterpret_cast<doublecomplex*>(dat), n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix*)
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
      GlobalLU_t gLU;
      zgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
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

#if __has_include("slu_cdefs.h")
  template<>
  struct SuperLUDenseMatChooser<std::complex<float> >
  {
    static void create(SuperMatrix *mat, int n, int m, std::complex<float> *dat, int n1,
                       Stype_t stype, Dtype_t dtype, Mtype_t mtype)
    {
      cCreate_Dense_Matrix(mat, n, m, reinterpret_cast< ::complex*>(dat), n1, stype, dtype, mtype);

    }

    static void destroy(SuperMatrix* /* mat */)
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
      GlobalLU_t gLU;
      cgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
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

  namespace Impl
  {
    template<class M>
    struct SuperLUVectorChooser
    {};

    template<typename T, typename A, int n, int m>
    struct SuperLUVectorChooser<BCRSMatrix<FieldMatrix<T,n,m>,A > >
    {
      /** @brief The type of the domain of the solver */
      using domain_type = BlockVector<
                              FieldVector<T,m>,
                              typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,m> > >;
      /** @brief The type of the range of the solver */
      using range_type  = BlockVector<
                              FieldVector<T,n>,
                              typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,n> > >;
    };

    template<typename T, typename A>
    struct SuperLUVectorChooser<BCRSMatrix<T,A> >
    {
      /** @brief The type of the domain of the solver */
      using domain_type = BlockVector<T, A>;
      /** @brief The type of the range of the solver */
      using range_type  = BlockVector<T, A>;
    };
  }

  /**
   * @brief SuperLu Solver
   *
   * Uses the well known <a href="http://crd.lbl.gov/~xiaoye/SuperLU/">SuperLU
   * package</a> to solve the
   * system.
   *
   * SuperLU supports single and double precision floating point and complex
   * numbers. Unfortunately these cannot be used at the same time.
   * Therefore users must set SUPERLU_NTYPE (0: float, 1: double,
   * 2: std::complex<float>, 3: std::complex<double>)
   * if the numeric type should be different from double.
   */
  template<typename M>
  class SuperLU
    : public InverseOperator<
          typename Impl::SuperLUVectorChooser<M>::domain_type,
          typename Impl::SuperLUVectorChooser<M>::range_type >
  {
    using T = typename M::field_type;
  public:
    /** @brief The matrix type. */
    using Matrix = M;
    using matrix_type = M;
    /** @brief The corresponding SuperLU Matrix type.*/
    typedef Dune::SuperLUMatrix<Matrix> SuperLUMatrix;
    /** @brief Type of an associated initializer class. */
    typedef SuperMatrixInitializer<Matrix> MatrixInitializer;
    /** @brief The type of the domain of the solver. */
    using domain_type = typename Impl::SuperLUVectorChooser<M>::domain_type;
    /** @brief The type of the range of the solver. */
    using range_type = typename Impl::SuperLUVectorChooser<M>::range_type;

    //! Category of the solver (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::Category::sequential;
    }

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


    /** @brief Constructs the SuperLU solver.
     *
     * @param matrix  The matrix of the system to solve.
     * @param config  ParameterTree containing solver parameters.
     *
     * ParameterTree Key | Meaning
     * ------------------|------------
     * verbose           | The verbosity level. default=false
     * reuseVector       | Reuse initially allocated vectors in apply. default=true
    */
    SuperLU(const Matrix& mat, const ParameterTree& config)
      : SuperLU(mat, config.get<bool>("verbose", false), config.get<bool>("reuseVector", true))
    {}

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
    void apply (domain_type& x, range_type& b, [[maybe_unused]] double reduction, InverseOperatorResult& res)
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
      return mat.nonzeroes();
    }

    template<class S>
    void setSubMatrix(const Matrix& mat, const S& rowIndexSet);

    void setVerbosity(bool v);

    /**
     * @brief free allocated space.
     * @warning later calling apply will result in an error.
     */
    void free();

    const char* name() { return "SuperLU"; }
  private:
    template<class Mat,class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;
    friend struct SeqOverlappingSchwarzAssemblerHelper<SuperLU<Matrix>,true>;

    SuperLUMatrix& getInternalMatrix() { return mat; }

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

  template<typename M>
  SuperLU<M>
  ::~SuperLU()
  {
    if(mat.N()+mat.M()>0)
      free();
  }

  template<typename M>
  void SuperLU<M>::free()
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

  template<typename M>
  SuperLU<M>
  ::SuperLU(const Matrix& mat_, bool verbose_, bool reusevector_)
    : work(0), lwork(0), first(true), verbose(verbose_),
      reusevector(reusevector_)
  {
    setMatrix(mat_);

  }
  template<typename M>
  SuperLU<M>::SuperLU()
    :    work(0), lwork(0),verbose(false),
      reusevector(false)
  {}
  template<typename M>
  void SuperLU<M>::setVerbosity(bool v)
  {
    verbose=v;
  }

  template<typename M>
  void SuperLU<M>::setMatrix(const Matrix& mat_)
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

  template<typename M>
  template<class S>
  void SuperLU<M>::setSubMatrix(const Matrix& mat_,
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

  template<typename M>
  void SuperLU<M>::decompose()
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

    typename GetSuperLUType<T>::float_type rpg, rcond, ferr=1e10, berr=1e10;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;

    StatInit(&stat);
    SuperLUSolveChooser<T>::solve(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
                                  &L, &U, work, lwork, &B, &X, &rpg, &rcond, &ferr,
                                  &berr, &memusage, &stat, &info);

    if(verbose) {
      dinfo<<"LU factorization: dgssvx() returns info "<< info<<std::endl;

      auto nSuperLUCol = static_cast<SuperMatrix&>(mat).ncol;

      if ( info == 0 || info == nSuperLUCol+1 ) {

        if ( options.PivotGrowth )
          dinfo<<"Recip. pivot growth = "<<rpg<<std::endl;
        if ( options.ConditionNumber )
          dinfo<<"Recip. condition number = %e\n"<< rcond<<std::endl;
        SCformat* Lstore = (SCformat *) L.Store;
        NCformat* Ustore = (NCformat *) U.Store;
        dinfo<<"No of nonzeros in factor L = "<< Lstore->nnz<<std::endl;
        dinfo<<"No of nonzeros in factor U = "<< Ustore->nnz<<std::endl;
        dinfo<<"No of nonzeros in L+U = "<< Lstore->nnz + Ustore->nnz - nSuperLUCol<<std::endl;
        QuerySpaceChooser<T>::querySpace(&L, &U, &memusage);
        dinfo<<"L\\U MB "<<memusage.for_lu/1e6<<" \ttotal MB needed "<<memusage.total_needed/1e6
             <<" \texpansions ";
        std::cout<<stat.expansions<<std::endl;

      } else if ( info > 0 && lwork == -1 ) {    // Memory allocation failed
        dinfo<<"** Estimated memory: "<< info - nSuperLUCol<<std::endl;
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

  template<typename M>
  void SuperLU<M>
  ::apply(domain_type& x, range_type& b, InverseOperatorResult& res)
  {
    if (mat.N() != b.dim())
      DUNE_THROW(ISTLError, "Size of right-hand-side vector b does not match the number of matrix rows!");
    if (mat.M() != x.dim())
      DUNE_THROW(ISTLError, "Size of solution vector x does not match the number of matrix columns!");
    if (mat.M()+mat.N()==0)
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
        ((DNformat*)B.Store)->nzval=&b[0];
        ((DNformat*)X.Store)->nzval=&x[0];
      }
    } else {
      SuperLUDenseMatChooser<T>::create(&rB, (int)mat.N(), 1,  reinterpret_cast<T*>(&b[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      SuperLUDenseMatChooser<T>::create(&rX, (int)mat.N(), 1,  reinterpret_cast<T*>(&x[0]), (int)mat.N(), SLU_DN, GetSuperLUType<T>::type, SLU_GE);
      mB = &rB;
      mX = &rX;
    }
    typename GetSuperLUType<T>::float_type rpg, rcond, ferr=1e10, berr;
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
    options.IterRefine=SLU_DOUBLE;

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

      auto nSuperLUCol = static_cast<SuperMatrix&>(mat).ncol;

      if ( info == 0 || info == nSuperLUCol+1 ) {

        if ( options.IterRefine ) {
          std::cout<<"Iterative Refinement: steps="
                   <<stat.RefineSteps<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
        }else
          std::cout<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
      } else if ( info > 0 && lwork == -1 ) {       // Memory allocation failed
        std::cout<<"** Estimated memory: "<< info - nSuperLUCol<<" bytes"<<std::endl;
      }

      if ( options.PrintStat ) StatPrint(&stat);
    }
    StatFree(&stat);
    if (!reusevector) {
      SUPERLU_FREE(rB.Store);
      SUPERLU_FREE(rX.Store);
    }
  }

  template<typename M>
  void SuperLU<M>
  ::apply(T* x, T* b)
  {
    if(mat.N()+mat.M()==0)
      DUNE_THROW(ISTLError, "Matrix of SuperLU is null!");

    SuperMatrix* mB = &B;
    SuperMatrix* mX = &X;
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
      mB = &rB;
      mX = &rX;
    }

    typename GetSuperLUType<T>::float_type rpg, rcond, ferr=1e10, berr;
    int info;
    mem_usage_t memusage;
    SuperLUStat_t stat;
    /* Initialize the statistics variables. */
    StatInit(&stat);

    options.IterRefine=SLU_DOUBLE;

    SuperLUSolveChooser<T>::solve(&options, &static_cast<SuperMatrix&>(mat), perm_c, perm_r, etree, &equed, R, C,
                                  &L, &U, work, lwork, mB, mX, &rpg, &rcond, &ferr, &berr,
                                  &memusage, &stat, &info);

    if(verbose) {
      dinfo<<"Triangular solve: dgssvx() returns info "<< info<<std::endl;

      auto nSuperLUCol = static_cast<SuperMatrix&>(mat).ncol;

      if ( info == 0 || info == nSuperLUCol+1 ) {  // Factorization has succeeded

        if ( options.IterRefine ) {
          dinfo<<"Iterative Refinement: steps="
               <<stat.RefineSteps<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
        }else
          dinfo<<" FERR="<<ferr<<" BERR="<<berr<<std::endl;
      } else if ( info > 0 && lwork == -1 ) {  // Memory allocation failed
        dinfo<<"** Estimated memory: "<< info - nSuperLUCol<<" bytes"<<std::endl;
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

  template<typename T, typename A>
  struct IsDirectSolver<SuperLU<BCRSMatrix<T,A> > >
  {
    enum { value=true};
  };

  template<typename T, typename A>
  struct StoresColumnCompressed<SuperLU<BCRSMatrix<T,A> > >
  {
    enum { value = true };
  };

  struct SuperLUCreator {
    template<class> struct isValidBlock : std::false_type{};
    template<int k> struct isValidBlock<Dune::FieldVector<double,k>> : std::true_type{};
    template<int k> struct isValidBlock<Dune::FieldVector<std::complex<double>,k>> : std::true_type{};
    template<typename TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const M& mat, const Dune::ParameterTree& config,
                std::enable_if_t<isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      int verbose = config.get("verbose", 0);
      return std::make_shared<Dune::SuperLU<M>>(mat,verbose);
    }

    // second version with SFINAE to validate the template parameters of SuperLU
    template<typename TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const M& /*mat*/, const Dune::ParameterTree& /*config*/,
      std::enable_if_t<!isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      DUNE_THROW(UnsupportedType,
        "Unsupported Type in SuperLU (only double and std::complex<double> supported)");
    }
  };
  template<> struct SuperLUCreator::isValidBlock<double> : std::true_type{};
  template<> struct SuperLUCreator::isValidBlock<std::complex<double>> : std::true_type{};

  DUNE_REGISTER_DIRECT_SOLVER("superlu", SuperLUCreator());
} // end namespace DUNE

// undefine macros from SuperLU's slu_util.h
#undef FIRSTCOL_OF_SNODE
#undef NO_MARKER
#undef NUM_TEMPV
#undef USER_ABORT
#undef USER_MALLOC
#undef SUPERLU_MALLOC
#undef USER_FREE
#undef SUPERLU_FREE
#undef CHECK_MALLOC
#undef SUPERLU_MAX
#undef SUPERLU_MIN
#undef L_SUB_START
#undef L_SUB
#undef L_NZ_START
#undef L_FST_SUPC
#undef U_NZ_START
#undef U_SUB
#undef TRUE
#undef FALSE
#undef EMPTY
#undef NODROP
#undef DROP_BASIC
#undef DROP_PROWS
#undef DROP_COLUMN
#undef DROP_AREA
#undef DROP_SECONDARY
#undef DROP_DYNAMIC
#undef DROP_INTERP
#undef MILU_ALPHA

#endif // HAVE_SUPERLU
#endif // DUNE_SUPERLU_HH
