// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SUPERLUFUNCTIONS_HH
#define DUNE_ISTL_SUPERLUFUNCTIONS_HH
#if HAVE_SUPERLU


#define int_t SUPERLU_INT_TYPE
#include "supermatrix.h"
#include "slu_util.h"
#undef int_t

#if __has_include("slu_sdefs.h")
extern "C" {
  extern void
  sgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
         char *, float *, float *, SuperMatrix *, SuperMatrix *,
         void *, int, SuperMatrix *, SuperMatrix *,
         float *, float *, float *, float *,
         GlobalLU_t*, mem_usage_t *, SuperLUStat_t *, int *);

  extern void
  sCreate_Dense_Matrix(SuperMatrix *, int, int, float *, int,
                       Stype_t, Dtype_t, Mtype_t);
  extern void
  sCreate_CompCol_Matrix(SuperMatrix *, int, int, int, float *,
                         int *, int *, Stype_t, Dtype_t, Mtype_t);
  extern int     sQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

  extern void    sPrint_CompCol_Matrix(char *, SuperMatrix *);
}
#endif

#if __has_include("slu_ddefs.h")
extern "C" {
  extern void
  dgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
         char *, double *, double *, SuperMatrix *, SuperMatrix *,
         void *, int, SuperMatrix *, SuperMatrix *,
         double *, double *, double *, double *,
         GlobalLU_t*, mem_usage_t *, SuperLUStat_t *, int *);

  extern void
  dCreate_CompCol_Matrix(SuperMatrix *, int, int, int, double *,
                         int *, int *, Stype_t, Dtype_t, Mtype_t);

  extern void
  dCreate_Dense_Matrix(SuperMatrix *, int, int, double *, int,
                       Stype_t, Dtype_t, Mtype_t);

  extern int     dQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

  extern void    dPrint_CompCol_Matrix(char *, SuperMatrix *);
}
#endif

#if __has_include("slu_cdefs.h")
#include "slu_scomplex.h"

extern "C" {
  extern void
  cgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
         char *, float *, float *, SuperMatrix *, SuperMatrix *,
         void *, int, SuperMatrix *, SuperMatrix *,
         float *, float *, float *, float *,
         GlobalLU_t*, mem_usage_t *, SuperLUStat_t *, int *);


  extern void
  cCreate_Dense_Matrix(SuperMatrix *, int, int, ::complex *, int,
                       Stype_t, Dtype_t, Mtype_t);


  extern void
  cCreate_CompCol_Matrix(SuperMatrix *, int, int, int, ::complex *,
                         int *, int *, Stype_t, Dtype_t, Mtype_t);

  extern int     cQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

  extern void    cPrint_CompCol_Matrix(char *, SuperMatrix *);
}
#endif

#if __has_include("slu_zdefs.h")
#include "slu_dcomplex.h"
extern "C" {
  extern void
  zgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
         char *, double *, double *, SuperMatrix *, SuperMatrix *,
         void *, int, SuperMatrix *, SuperMatrix *,
         double *, double *, double *, double *,
         GlobalLU_t*, mem_usage_t *, SuperLUStat_t *, int *);


  extern void
  zCreate_CompCol_Matrix(SuperMatrix *, int, int, int, doublecomplex *,
                         int *, int *, Stype_t, Dtype_t, Mtype_t);

  extern void
  zCreate_Dense_Matrix(SuperMatrix *, int, int, doublecomplex *, int,
                       Stype_t, Dtype_t, Mtype_t);

  extern int     zQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

  extern void    zPrint_CompCol_Matrix(char *, SuperMatrix *);
}
#endif


#endif
#endif
