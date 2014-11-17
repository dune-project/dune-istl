// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_LDL_HH
#define DUNE_LDL_HH

#if HAVE_LDL

#include <iostream>
#include<complex>
#include<type_traits>

#ifdef __cplusplus
extern "C"
{
#include "ldl.h"
#include "amd.h"
}
#endif

#include<dune/common/exceptions.hh>
#include<dune/common/unused.hh>

#include<dune/istl/solvers.hh>
#include<dune/istl/solvertype.hh>
#include<dune/istl/colcompmatrix.hh>

namespace Dune {
  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Marco Agnese, Andrea Sacconi
   * @brief Class for using LDL with ISTL matrices.
   */

  // forward declaration
  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /**
   * @brief The %LDL direct sparse solver for matrices of type Matrix
   * Details on LDL can be found on http://www.cise.ufl.edu/research/sparse/ldl/
   * \note This will only work if dune-istl has been configured to use LDL
   */
  template<typename MT>
  class LDL
  {
    public:
    /** @brief The matrix type. */
    typedef MT Matrix;

    /** @brief The column-compressed matrix type.*/
    typedef ColCompMatrix<Matrix> LDLMatrix;

    /**
     * @brief Construct a solver object.
     * @param mat the matrix to solve for
     * @param verbose 0 or 1 set the verbosity level, defaults to 0
     */
    LDL(const Matrix& mat, int verbose=0) : isloaded_(false), verbose_(verbose)
    {
      setMatrix(mat);
    }

    /**
     * @brief Constructor for compatibility with SuperLU standard constructor
     * @param mat the matrix to solve for
     * @param verbose 0 or 1 set the verbosity level, defaults to 0
     */
    LDL(const Matrix& mat, int verbose, bool) : isloaded_(false), verbose_(verbose)
    {
      setMatrix(mat);
    }

    /** @brief Default constructor. */
    LDL() : isloaded_(false), verbose_(0)
    {}

    /** @brief Default constructor. */
    ~LDL()
    {
      if ((mat_.N() + mat_.M() > 0) || isloaded_)
        free();
    }

    /** @brief Solve the system Ax=b. */
    template<class DT,class RT>
    void apply(DT& x, const RT& b, InverseOperatorResult& res)
    {
      const int dimMat(mat_.N());
      ldl_perm (dimMat, Y_, reinterpret_cast<double*>(&b[0]), P_);
      ldl_lsolve(dimMat, Y_, Lp_, Li_, Lx_);
      ldl_dsolve(dimMat, Y_, D_);
      ldl_ltsolve(dimMat, Y_, Lp_, Li_, Lx_);
      ldl_permt (dimMat, reinterpret_cast<double*>(&x[0]), Y_, P_);
      // this is a direct solver
      res.iterations = 1;
      res.converged = true;
    }

    /** @brief Solve the system Ax=b. */
    template<class DT,class RT>
    inline void apply(DT& x, const RT& b, double reduction, InverseOperatorResult& res)
    {
      DUNE_UNUSED_PARAMETER(reduction);
      apply(x,b,res);
    }

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& mat)
    {
      if ((mat_.N() + mat_.M() > 0) || isloaded_)
        free();
      mat_ = mat;

      // set correct dimension for additional vectors (only those before symbolic)
      const int dimMat(mat_.N());
      D_ = new double [dimMat];
      Y_ = new double [dimMat];
      Lp_ = new int [dimMat + 1];
      Parent_ = new int [dimMat];
      Lnz_ = new int [dimMat];
      Flag_ = new int [dimMat];
      Pattern_ = new int [dimMat];
      P_ = new int [dimMat];
      Pinv_ = new int [dimMat];

      decompose();
    }

    /**
     * @brief Sets the verbosity level for the solver.
     * @param v verbosity level: 0 only error messages, 1 a bit of statistics.
     */
    inline void setVerbosity(int v)
    {
      verbose_=v;
    }

    /**
     * @brief Free allocated space.
     * @warning Later calling apply will result in an error.
     */
    void free()
    {
      delete [] D_;
      delete [] Y_;
      delete [] Lp_;
      delete [] Parent_;
      delete [] Lnz_;
      delete [] Flag_;
      delete [] Pattern_;
      delete [] Lx_;
      delete [] Li_;
      delete [] P_;
      delete [] Pinv_;
      mat_.free();
      isloaded_ = false;
    }

    /** @brief Get method name. */
    inline const char* name()
    {
      return "LDL";
    }

    /**
     * @brief Get factorization diagonal matrix D.
     * @warning It is up to the user to preserve consistency when modifyng it.
     */
    inline double* getD(void)
    {
      return D_;
    }

    /**
     * @brief Get factorization Lp.
     * @warning It is up to the user to preserve consistency when modifyng it.
     */
    inline int* getLp(void)
    {
      return Lp_;
    }

    /**
     * @brief Get factorization Li.
     * @warning It is up to the user to preserve consistency when modifyng it.
     */
    inline int* getLi(void)
    {
      return Li_;
    }

    /**
     * @brief Get factorization Lx.
     * @warning It is up to the user to preserve consistency when modifyng it.
     */
    inline double* getLx(void)
    {
      return Lx_;
    }

    private:
    template<class M,class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;

    friend struct SeqOverlappingSchwarzAssemblerHelper<LDL<Matrix>,true>;

    /** @brief Computes the LDL decomposition. */
    void decompose()
    {
      const int dimMat(mat_.N());
      double Info [AMD_INFO];
      if (amd_order (dimMat, mat_.getColStart(), mat_.getRowIndex(), P_, (double *) NULL, Info) < AMD_OK)
        std::cout<<"WARNING: call to AMD failed."<<std::endl;
      if(verbose_ > 0)
        amd_info (Info);
      // compute the symbolic factorisation
      ldl_symbolic(dimMat, mat_.getColStart(), mat_.getRowIndex(), Lp_, Parent_, Lnz_, Flag_, P_, Pinv_);
      // initialise those entries of additionalVectors_ whose dimension is known only now
      Lx_ = new double [Lp_[dimMat]];
      Li_ = new int [Lp_[dimMat]];
      // compute the numeric factorisation
      const int rank(ldl_numeric(dimMat, mat_.getColStart(), mat_.getRowIndex(), mat_.getValues(),
                                 Lp_, Parent_, Lnz_, Li_, Lx_, D_, Y_, Pattern_, Flag_, P_, Pinv_));
      if(rank!=dimMat)
        std::cout<<"WARNING: matrix is singular."<<std::endl;
    }

    LDLMatrix mat_;
    bool isloaded_;
    int verbose_;
    int* Lp_;
    int* Parent_;
    int* Lnz_;
    int* Flag_;
    int* Pattern_;
    int* P_;
    int* Pinv_;
    double* D_;
    double* Y_;
    double* Lx_;
    int* Li_;
  };

  template<typename MT>
  struct IsDirectSolver<LDL<MT> >
  {
    enum {value=true};
  };

  template<typename MT>
  struct StoresColumnCompressed<LDL<MT> >
  {
    enum {value=true};
  };
}

#endif //HAVE_LDL
#endif //DUNE_LDL_HH
