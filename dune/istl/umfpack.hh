// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_UMFPACK_HH
#define DUNE_UMFPACK_HH

#if HAVE_UMFPACK

#include<complex>
#include<type_traits>

#include<umfpack.h>

#include<dune/common/exceptions.hh>
#include<dune/common/fmatrix.hh>
#include<dune/common/fvector.hh>
#include<dune/istl/bcrsmatrix.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/solvertype.hh>

#include"colcompmatrix.hh"


namespace Dune {
  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Dominic Kempf
   * @brief Classes for using UMFPack with ISTL matrices.
   */

  // FORWARD DECLARATIONS
  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /** @brief use the UMFPack Package to directly solve linear systems
   * @tparam Matrix the matrix type defining the system
   * Details on UMFPack are to be found on
   * http://www.cise.ufl.edu/research/sparse/umfpack/
   */
  template<class Matrix>
  class UMFPack
  {};

  // wrapper class for C-Function Calls in the backend. Choose the right function namespace
  // depending on the template parameter used.
  template<typename T>
  struct UMFPackMethodChooser
  {};

  template<>
  struct UMFPackMethodChooser<double>
  {
    template<typename... A>
    static void defaults(A... args)
    {
      umfpack_di_defaults(args...);
    }
    template<typename... A>
    static void free_numeric(A... args)
    {
      umfpack_di_free_numeric(args...);
    }
    template<typename... A>
    static void free_symbolic(A... args)
    {
      umfpack_di_free_symbolic(args...);
    }
    template<typename... A>
    static int load_numeric(A... args)
    {
      return umfpack_di_load_numeric(args...);
    }
    template<typename... A>
    static void numeric(A... args)
    {
      umfpack_di_numeric(args...);
    }
    template<typename... A>
    static void report_info(A... args)
    {
      umfpack_di_report_info(args...);
    }
    template<typename... A>
    static void report_status(A... args)
    {
      umfpack_di_report_status(args...);
    }
    template<typename... A>
    static int save_numeric(A... args)
    {
      return umfpack_di_save_numeric(args...);
    }
    template<typename... A>
    static void solve(A... args)
    {
      umfpack_di_solve(args...);
    }
    template<typename... A>
    static void symbolic(A... args)
    {
      umfpack_di_symbolic(args...);
    }
  };

  template<>
  struct UMFPackMethodChooser<std::complex<double> >
  {
    template<typename... A>
    static void defaults(A... args)
    {
      umfpack_zi_defaults(args...);
    }
    template<typename... A>
    static void free_numeric(A... args)
    {
      umfpack_zi_free_numeric(args...);
    }
    template<typename... A>
    static void free_symbolic(A... args)
    {
      umfpack_zi_free_symbolic(args...);
    }
    template<typename... A>
    static int load_numeric(A... args)
    {
      return umfpack_zi_load_numeric(args...);
    }
    template<typename... A>
    static void numeric(const int* cs, const int* ri, const double* val, A... args)
    {
      umfpack_zi_numeric(cs,ri,val,NULL,args...);
    }
    template<typename... A>
    static void report_info(A... args)
    {
      umfpack_zi_report_info(args...);
    }
    template<typename... A>
    static void report_status(A... args)
    {
      umfpack_zi_report_status(args...);
    }
    template<typename... A>
    static int save_numeric(A... args)
    {
      return umfpack_zi_save_numeric(args...);
    }
    template<typename... A>
    static void solve(int m, const int* cs, const int* ri, std::complex<double>* val, double* x, const double* b,A... args)
    {
      const double* cval = reinterpret_cast<const double*>(val);
      umfpack_zi_solve(m,cs,ri,cval,NULL,x,NULL,b,NULL,args...);
    }
    template<typename... A>
    static void symbolic(int m, int n, const int* cs, const int* ri, const double* val, A... args)
    {
      umfpack_zi_symbolic(m,n,cs,ri,val,NULL,args...);
    }
  };

  /** @brief use the UMFPack Package to directly solve linear systems
   *
   * Specialization for the Dune::BCRSMatrix. UMFPack will always go double
   * precision and supports complex numbers
   * too (use std::complex<double> for that).
   */
  template<typename T, typename A, int n, int m>
  class UMFPack<BCRSMatrix<FieldMatrix<T,n,m>,A > >
      : public InverseOperator<
          BlockVector<FieldVector<T,m>,
              typename A::template rebind<FieldVector<T,m> >::other>,
          BlockVector<FieldVector<T,n>,
              typename A::template rebind<FieldVector<T,n> >::other> >
  {
    public:
    /** @brief The matrix type. */
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> matrix_type;
    /** @brief The corresponding SuperLU Matrix type.*/
    typedef Dune::ColCompMatrix<Matrix> UMFPackMatrix;
    /** @brief Type of an associated initializer class. */
    typedef ColCompMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A> > MatrixInitializer;
    /** @brief The type of the domain of the solver. */
    typedef Dune::BlockVector<
        FieldVector<T,m>,
        typename A::template rebind<FieldVector<T,m> >::other> domain_type;
    /** @brief The type of the range of the solver. */
    typedef Dune::BlockVector<
        FieldVector<T,n>,
        typename A::template rebind<FieldVector<T,n> >::other> range_type;

    /** @brief construct a solver object from a BCRSMatrix
     *  @param mat_ the matrix to solve for
     *  @param verbose [0..2] set the verbosity level, defaults to 0
     */
    UMFPack(const Matrix& mat_, int verbose=0) : mat_is_loaded(false)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in UMFPack (only double and std::complex<double> supported)");
      Caller::defaults(UMF_Control);
      setVerbosity(verbose);
      setMatrix(mat_);
    }

    /** @brief Constructor for compatibility with SuperLU standard constructor
     * @param mat_ the matrix to solve for
     * @param verbose [0..2] set the verbosity level, defaults to 0
     */
    UMFPack(const Matrix& mat_, int verbose, bool) : mat_is_loaded(false)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in UMFPack (only double and std::complex<double> supported)");
      Caller::defaults(UMF_Control);
      setVerbosity(verbose);
      setMatrix(mat_);
    }

    /** @brief default constructor
     */
    UMFPack() : mat_is_loaded(false), verbose(0)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in UMFPack (only double and std::complex<double> supported)");
      Caller::defaults(UMF_Control);
    }

    /** @brief try loading a decomposition from file and do a decomposition if unsuccesful
     * @param mat_ the matrix to decompose when no decoposition file found
     * @param file the decomposition file
     * @param verbose the verbosity level
     * use saveDecomposition(char* file) for manually storing a decomposition. This constructor
     * will decompose mat_ and store the result to file if no file wasnt found in the first place.
     * Thus, if you always use this you will only compute the decomposition once (and when you manually
     * deleted the decomposition file).
     */
    UMFPack(const Matrix& mat_, const char* file, int verbose=0)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in UMFPack (only double and std::complex<double> supported)");
      Caller::defaults(UMF_Control);
      setVerbosity(verbose);
      int errcode = Caller::load_numeric(&UMF_Numeric, const_cast<char*>(file));
      if ((errcode == UMFPACK_ERROR_out_of_memory) || (errcode == UMFPACK_ERROR_file_IO))
      {
        mat_is_loaded = false;
        setMatrix(mat_);
        saveDecomposition(file);
      }
      else
      {
        mat_is_loaded = true;
        std::cout << "UMFPack decomposition succesfully loaded from " << file << std::endl;
      }
    }

    /** @brief try loading a decomposition from file
     * @param file the decomposition file
     * @param verbose the verbosity level
     * throws an exception when not being able to load the file. Doesnt need knowledge of the
     * actual matrix!
     */
    UMFPack(const char* file, int verbose=0)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in UMFPack (only double and std::complex<double> supported)");
      Caller::defaults(UMF_Control);
      int errcode = Caller::load_numeric(&UMF_Numeric, const_cast<char*>(file));
      if (errcode == UMFPACK_ERROR_out_of_memory)
        DUNE_THROW(Dune::Exception, "ran out of memory while loading UMFPack decomposition");
      if (errcode == UMFPACK_ERROR_file_IO)
        DUNE_THROW(Dune::Exception, "IO error while loading UMFPack decomposition");
      mat_is_loaded = true;
      std::cout << "UMFPack decomposition succesfully loaded from " << file << std::endl;
      setVerbosity(verbose);
    }

    virtual ~UMFPack()
    {
      if ((mat.N() + mat.M() > 0) || (mat_is_loaded))
        free();
    }

    /**
     *  \copydoc InverseOperator::apply(X&, Y&, InverserOperatorResult&)
     */
    virtual void apply(domain_type& x, range_type& b, InverseOperatorResult& res)
    {
      double UMF_Apply_Info[UMFPACK_INFO];
      Caller::solve(UMFPACK_A,
                    mat.getColStart(),
                    mat.getRowIndex(),
                    mat.getValues(),
                    reinterpret_cast<double*>(&x[0]),
                    reinterpret_cast<double*>(&b[0]),
                    UMF_Numeric,
                    UMF_Control,
                    UMF_Apply_Info);

      //this is a direct solver
      res.iterations = 1;
      res.converged = true;

      printOnApply(UMF_Apply_Info);
    }

    /**
     *  \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
     */
    virtual void apply (domain_type& x, range_type& b, double reduction, InverseOperatorResult& res)
    {
      apply(x,b,res);
    }

    /**
     * @brief additional apply method with c-arrays in analogy to superlu
     * @param x solution array
     * @param b rhs array
     */
    void apply(T* x, T* b)
    {
      double UMF_Apply_Info[UMFPACK_INFO];
      Caller::solve(UMFPACK_A,
                    mat.getColStart(),
                    mat.getRowIndex(),
                    mat.getValues(),
                    x,
                    b,
                    UMF_Numeric,
                    UMF_Control,
                    UMF_Apply_Info);
      printOnApply(UMF_Apply_Info);
    }

    /** @brief saves a decomposition to a file
     * @param file the filename to save to
     */
    void saveDecomposition(const char* file)
    {
      int errcode = Caller::save_numeric(UMF_Numeric, const_cast<char*>(file));
      if (errcode != UMFPACK_OK)
        DUNE_THROW(Dune::Exception,"IO ERROR while trying to save UMFPack decomposition");
    }

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& _mat)
    {
      if ((mat.N() + mat.M() > 0) || (mat_is_loaded))
        free();
      mat = _mat;
      decompose();
    }

    template<class S>
    void setSubMatrix(const Matrix& _mat, const S& rowIndexSet)
    {
      if ((mat.N() + mat.M() > 0) || (mat_is_loaded))
        free();
      mat.setMatrix(_mat,rowIndexSet);
      decompose();
    }

    /** @brief sets the verbosity level for the UMFPack solver
     * @param v verbosity level
     * The following levels are implemented:
     * 0 - only error messages
     * 1 - a bit of statistics on decomposition and solution
     * 2 - lots of statistics on decomposition and solution
     */
    void setVerbosity(int v)
    {
      verbose = v;
      // set the verbosity level in UMFPack
      if (verbose == 0)
        UMF_Control[UMFPACK_PRL] = 1;
      if (verbose == 1)
        UMF_Control[UMFPACK_PRL] = 2;
      if (verbose == 2)
        UMF_Control[UMFPACK_PRL] = 4;
    }

    /**
     * @brief free allocated space.
     * @warning later calling apply will result in an error.
     */
    void free()
    {
      if (!mat_is_loaded)
      {
        Caller::free_symbolic(&UMF_Symbolic);
        mat.free();
      }
      Caller::free_numeric(&UMF_Numeric);
      mat_is_loaded = false;
    }

    private:
    typedef typename Dune::UMFPackMethodChooser<T> Caller;

    template<class M,class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;
    friend struct SeqOverlappingSchwarzAssemblerHelper<UMFPack<Matrix>,true>;

    /** @brief computes the LU Decomposition */
    void decompose()
    {
      double UMF_Decomposition_Info[UMFPACK_INFO];
      Caller::symbolic(static_cast<int>(mat.N()),
                       static_cast<int>(mat.N()),
                       mat.getColStart(),
                       mat.getRowIndex(),
                       reinterpret_cast<double*>(mat.getValues()),
                       &UMF_Symbolic,
                       UMF_Control,
                       UMF_Decomposition_Info);
      Caller::numeric(mat.getColStart(),
                      mat.getRowIndex(),
                      reinterpret_cast<double*>(mat.getValues()),
                      UMF_Symbolic,
                      &UMF_Numeric,
                      UMF_Control,
                      UMF_Decomposition_Info);
      Caller::report_status(UMF_Control,UMF_Decomposition_Info[UMFPACK_STATUS]);
      if (verbose == 1)
      {
        std::cout << "[UMFPack Decomposition]" << std::endl;
        std::cout << "Wallclock Time taken: " << UMF_Decomposition_Info[UMFPACK_NUMERIC_WALLTIME] << " (CPU Time: " << UMF_Decomposition_Info[UMFPACK_NUMERIC_TIME] << ")" << std::endl;
        std::cout << "Flops taken: " << UMF_Decomposition_Info[UMFPACK_FLOPS] << std::endl;
        std::cout << "Peak Memory Usage: " << UMF_Decomposition_Info[UMFPACK_PEAK_MEMORY]*UMF_Decomposition_Info[UMFPACK_SIZE_OF_UNIT] << " bytes" << std::endl;
        std::cout << "Condition number estimate: " << 1./UMF_Decomposition_Info[UMFPACK_RCOND] << std::endl;
        std::cout << "Numbers of non-zeroes in decomposition: L: " << UMF_Decomposition_Info[UMFPACK_LNZ] << " U: " << UMF_Decomposition_Info[UMFPACK_UNZ] << std::endl;
      }
      if (verbose == 2)
      {
        Caller::report_info(UMF_Control,UMF_Decomposition_Info);
      }
    }

    void printOnApply(double* UMF_Info)
    {
      Caller::report_status(UMF_Control,UMF_Info[UMFPACK_STATUS]);
      if (verbose > 0)
      {
        std::cout << "[UMFPack Solve]" << std::endl;
        std::cout << "Wallclock Time: " << UMF_Info[UMFPACK_SOLVE_WALLTIME] << " (CPU Time: " << UMF_Info[UMFPACK_SOLVE_TIME] << ")" << std::endl;
        std::cout << "Flops Taken: " << UMF_Info[UMFPACK_SOLVE_FLOPS] << std::endl;
        std::cout << "Iterative Refinement steps taken: " << UMF_Info[UMFPACK_IR_TAKEN] << std::endl;
        std::cout << "Error Estimate: " << UMF_Info[UMFPACK_OMEGA1] << " resp. " << UMF_Info[UMFPACK_OMEGA2] << std::endl;
      }
    }

    UMFPackMatrix mat;
    bool mat_is_loaded;
    int verbose;
    void *UMF_Symbolic;
    void *UMF_Numeric;
    double UMF_Control[UMFPACK_CONTROL];
  };

  template<typename T, typename A, int n, int m>
  struct IsDirectSolver<UMFPack<BCRSMatrix<FieldMatrix<T,n,m>,A> > >
  {
    enum { value=true};
  };

  template<typename T, typename A, int n, int m>
  struct StoresColumnCompressed<UMFPack<BCRSMatrix<FieldMatrix<T,n,m>,A> > >
  {
    enum { value = true };
  };
}

#endif //HAVE_UMFPACK

#endif //DUNE_UMFPACK_HH
