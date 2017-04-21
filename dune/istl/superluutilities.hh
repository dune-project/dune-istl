// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SUPERLU_UTILITIES_HH
#define DUNE_ISTL_SUPERLU_UTILITIES_HH

#if HAVE_SUPERLU

#include "superlufunctions.hh"
#include "supermatrix.hh"

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
   * @brief Classes for using the SuperLU LU solver and the SuperLU ILU preconditioner with ISTL matrices.
   * Most of them are there to choose the right SuperLU function
   * given the arithmetic type as template argument
   */

  //! Choosing the right driver function for the LU solver
  template<class T>
  struct SuperLUSolveChooser
  {};

  //! Choosing the right driver function for the ILU preconditioner
  template<class T>
  struct SuperLUILUSolveChooser
  {};

  //! Choosing the right dense matrix type create/destroy function
  template<class T>
  struct SuperLUDenseMatChooser
  {};

  //! Choosing the right function to query the memory usage
  //! \todo Can this be removed or should it replace the QuerySpaceChooser?
  template<class T>
  struct SuperLUQueryChooser
  {};

  //! Choosing the right function to query the memory usage
  template<class T>
  struct QuerySpaceChooser
  {};

#if HAVE_SLU_SDEFS_H
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
  struct SuperLUILUSolveChooser<float>
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      float *rpg, float *rcond,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      sgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             &gLU, memusage, stat, info);
#else
      sgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             memusage, stat, info);
#endif
    }
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
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      sgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
#else
      sgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
#endif
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

#endif // HAVE_SLU_SDEFS_H

#if HAVE_SLU_DDEFS_H

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
  struct SuperLUILUSolveChooser<double>
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, double *R, double *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      double *rpg, double *rcond,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      dgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             &gLU, memusage, stat, info);
#else
      dgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             memusage, stat, info);
#endif
    }
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
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      dgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
#else
      dgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
#endif
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
#endif // HAVE_SLU_DDEFS_H

#if HAVE_SLU_ZDEFS_H
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
  struct SuperLUILUSolveChooser<std::complex<double> >
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, double *R, double *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      double *rpg, double *rcond,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      zgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             &gLU, memusage, stat, info);
#else
      zgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             memusage, stat, info);
#endif
    }
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
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      zgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
#else
      zgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
#endif
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
#endif // HAVE_SLU_ZDEFS_H

#if HAVE_SLU_CDEFS_H
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
  struct SuperLUILUSolveChooser<std::complex<float> >
  {
    static void solve(superlu_options_t *options, SuperMatrix *mat, int *perm_c, int *perm_r, int *etree,
                      char *equed, float *R, float *C, SuperMatrix *L, SuperMatrix *U,
                      void *work, int lwork, SuperMatrix *B, SuperMatrix *X,
                      float *rpg, float *rcond,
                      mem_usage_t *memusage, SuperLUStat_t *stat, int *info)
    {
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      cgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             &gLU, memusage, stat, info);
#else
      cgsisx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond,
             memusage, stat, info);
#endif
    }
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
#if SUPERLU_MIN_VERSION_5
      GlobalLU_t gLU;
      cgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             &gLU, memusage, stat, info);
#else
      cgssvx(options, mat, perm_c, perm_r, etree, equed, R, C,
             L, U, work, lwork, B, X, rpg, rcond, ferr, berr,
             memusage, stat, info);
#endif
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
#endif // HAVE_SLU_CDEFS_H

} // end namespace Dune

#endif // HAVE_SUPERLU
#endif // DUNE_ISTL_SUPERLU_UTILITIES_HH
