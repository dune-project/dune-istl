// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_BLAS_KERNELS_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLAS_KERNELS_HH

/** \file
 * \brief C++ wrapper for BLAS kernels
 */


#include <type_traits>
#include <complex>

#include <dune/common/typeutilities.hh>
#include <dune/common/exceptions.hh>

#if HAVE_BLAS
#define SGEMM_FORTRAN FC_FUNC (sgemm, SGEMM)
#define DGEMM_FORTRAN FC_FUNC (dgemm, DGEMM)
#define CGEMM_FORTRAN FC_FUNC (cgemm, CGEMM)
#define ZGEMM_FORTRAN FC_FUNC (zgemm, ZGEMM)

extern "C" {
  extern void SGEMM_FORTRAN(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                            const float * alpha, const float * A, const int * lda,
                            const float * B, const int * ldb, const float * beta,
                            float * C, const int * ldc);
  extern void DGEMM_FORTRAN(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                            const double * alpha, const double * A, const int * lda,
                            const double * B, const int * ldb, const double * beta,
                            double * C, const int * ldc);
  extern void CGEMM_FORTRAN(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                            const void* alpha, const void* A, const int * lda,
                            const void* B, const int * ldb, const void* beta,
                            void* C, const int * ldc);
  extern void ZGEMM_FORTRAN(const char * transa, const char * transb, const int * m, const int * n, const int * k,
                            const void* alpha, const void* A, const int * lda,
                            const void* B, const int * ldb, const void* beta,
                            void* C, const int * ldc);
}

namespace Dune {
  namespace BLAS {
      template<class T>
      std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                       std::is_same_v<T, std::complex<float>> ||
                       std::is_same_v<T, std::complex<double>>>
      /*void*/ gemm(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                    const T* alpha, const T* A, const int* lda,
                    const T* B, const int* ldb, const T* beta,
                    T* C, const int* ldc){
        if constexpr(std::is_same_v<T,float>){
          SGEMM_FORTRAN(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        if constexpr(std::is_same_v<T,double>){
          DGEMM_FORTRAN(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        if constexpr(std::is_same_v<T,std::complex<float>>){
          CGEMM_FORTRAN(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        if constexpr(std::is_same_v<T,std::complex<double>>){
          ZGEMM_FORTRAN(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
      }
  }
}
#endif // HAVE_BLAS

#endif
