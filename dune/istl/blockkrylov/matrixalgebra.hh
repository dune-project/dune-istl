// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_MATRIXALGEBRA_HH
#define DUNE_ISTL_BLOCKKRYLOV_MATRIXALGEBRA_HH

/** \file \brief Provides an implementation of *-subalgebras of
    \f$K^{s\times s}\f$ for the generic implementation of block Krylov methods.
 */

#include <random>

#include <dune/common/simd/loop.hh>
#include <dune/common/filledarray.hh>
#include <dune/common/parallel/mpidata.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/transpose.hh>

#include "blas.hh"

namespace Dune {

#ifndef DOXYGEN
  namespace {
    // computes fused-multiply add. Necessary for interleaving operations for
    // LoopSIMD to enable proper compiler optimizations. See dune-common#223
    template<class T>
    T fma(Simd::Scalar<T> alpha, T x, T y){
      return alpha*x + y;
    }

    template<class T, size_t S, size_t A>
    LoopSIMD<T,S,A> fma(Simd::Scalar<T> alpha, const LoopSIMD<T,S,A>& x, const LoopSIMD<T,S,A>& y){
      LoopSIMD<T,S,A> out;
      for(size_t i=0; i<S;++i){
        out[i] = fma(alpha, x[i], y[i]);
      }
      return out;
    }

    // computes c += a^t b
    template<class T, size_t K>
    void mtm(const std::array<T, K>& a,
             const std::array<T, K>& b,
             std::array<T, Simd::lanes<T>()>& c){
      for(size_t i=0; i<K; ++i){
        for(size_t j=0; j<Simd::lanes<T>(); ++j){
          c[j] = fma(Simd::lane(j, a[i]),b[i],c[j]);
        }
      }
    }

    // computes c += a b
    template<class T, size_t K>
    void mm(const std::array<T, K>& a,
            const std::array<T, Simd::lanes<T>()>& b,
            std::array<T, K>& c){
      for(size_t i=0; i<Simd::lanes<T>(); ++i){
        for(size_t j=0; j<K; ++j){
          c[j] = fma(Simd::lane(i, a[j]),b[i], c[j]);
        }
      }
    }

    constexpr size_t LOOPSIMD_TILEROWS = 6;

    template<class T> struct isLoopSIMD : std::false_type {};
    template<class T, size_t S, size_t A> struct isLoopSIMD<LoopSIMD<T,S,A>> : std::true_type {};


    template<class LS, size_t P>
    constexpr bool LoopSIMDcheck(){
      if constexpr (isLoopSIMD<LS>::value){
        if constexpr (P%Simd::lanes<typename LS::value_type>() == 0
                      && Simd::lanes<typename LS::value_type>() > 1){
          return true;
        }
      }
      return false;
    }

    // computes the cholesky factor of a s.p.d. matrix.
    // A breakdown is indicated by zero diagonal entries
    template<class T, int N>
    std::vector<size_t> cholesky_factorize(FieldMatrix<T, N>& mat){
      using std::sqrt, std::real;
      std::vector<size_t> dependend_columns = {};
      for(size_t i=0; i<N;++i){
        for(size_t j=0; j<=i; ++j){
          T sum = mat[i][j];
          for(size_t k=0;k<j;++k)
            sum -= mat[i][k]*conjugateComplex(mat[j][k]);
          if(i > j){
            if(mat[j][j] != 0)
              mat[i][j] = sum/mat[j][j];
            else
              mat[i][j] = 0;
          }else{ // i == j
            if(real(sum) > 100.0*real(mat[i][j]*std::numeric_limits<T>::epsilon())){
              mat[i][i] = sqrt(sum);
            }else{
              dependend_columns.push_back(i);
              mat[i][i] = 0.0;  // indicates breakdown
            }
          }
        }
        for(size_t j=i+1;j<N;++j){
          mat[i][j] = 0.0;
        }
      }
      return dependend_columns;
    }
  }
#endif

  /** @defgroup ISTL_Blockkrylov Block Krylov Solvers
      @ingroup ISTL_Solver
      @brief Block Krylov methods for solving linear systems with multiple right-hand
      sides.

      These methods are build up on the block Krylov framework by
      Frommer et al. that parametrizes the block Krylov space by a generic
      *-subalgebra of the \f$s \times s\f$ matrices. The practical relevant
      *-subsubalgebra is the `ParallelMatrixAlgebra`. It is parametrized by a
      parameter `P` that determines the block size. `P` must be a divider of the
      numbers of right-hand sides \f$s\f$. The larger the block size the faster
      is the convergence in terms of iterations. However the building blocks
      scale like \f$\mathcal{O}(p^2)\f$ for large `P`. For small `P` the runtime
      is constant due to the fact that the kernels are memory-bound. For good
      performance `P` should also be a multiple of the hardware SIMD-width. For
      a modern architecture like Intel Skylake a good choice for `P` is between
      32 and 64.

     See
     - Frommer, A., Lund, K., & Szyld, D. B. (2017). Block Krylov subspace methods for functions of matrices.
     - Frommer, A., Lund, K., & Szyld, D. B. (2020). Block Krylov subspace methods for functions of matrices II: Modified block FOM. SIAM Journal on Matrix Analysis and Applications, 41(2), 804-837.
     - Dreier, N. (2020). Hardware-Oriented Krylov Methods for High-Performance Computing. PhD Thesis.
  */

  /** @addtogroup ISTL_Blockkrylov
      @{
  */
  /**
     @class ParallelMatrixAlgebra
     @brief Implementation of the block parallel matrix algebra

     Implements the subalgebra of S x S matrices where only the P x P diagonal
     blocks are allowed to be nonzero.  Practically, the purpose of this class
     is to balance the computitional effort and the memory throughput.

     To make use of this effect it is important to use performant
     implementations of the costly kernels (i.e. axpy and dot operations). For
     nested SIMD-types (i.e. LoopSIMD of a SIMD-type,
     e.g. `LoopSIMD<Vc::Vector<double>, 8>`) we provide a tuned
     implementation. Alternatively, this routines are implemented by using the
     BLAS gemm kernel. The implementation is selected by the following
     priorities and conditions.

     | Overload                |  Priority  | Condition                                                                       |
     |-------------------------|------------|---------------------------------------------------------------------------------|
     | LoopSIMD specialization | 7          | see `LoopSIMDcheck`                                                             |
     | non-block spec.         | 9          | `P == 1`                                                                        |
     | full-block spec.        | 4          | `P == S`                                                                        |
     | fall-back               | 0          | none                                                                            |
     | BLAS gemm               | 5          | BLAS avail. can be adjusted via cmake variable `DUNE_BLOCKKRYLOV_BLAS_PRIORITY` |

     \tparam X the vector space the matrixalgebra operates on
     \tparam P block size
  **/

  template<class X, size_t P>
  class ParallelMatrixAlgebra{
  public:
    using vector_type = X;
    using field_type = typename X::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
    using scalar_type = Simd::Scalar<field_type>;
    using real_scalar_type = Simd::Scalar<real_type>;
    static constexpr size_t N = Simd::lanes<field_type>();
    static constexpr size_t Q = N/P;
    static_assert(N%P==0, "The block width P must be a divider of the SIMD width N.");
    std::array<FieldMatrix<scalar_type, P, P>, Q> value_ = filledArray<Q>(Dune::FieldMatrix<scalar_type, P, P>(0.0));
    ParallelMatrixAlgebra() = default;
    ParallelMatrixAlgebra(const ParallelMatrixAlgebra&) = default;
    ParallelMatrixAlgebra& operator=(const ParallelMatrixAlgebra&) = default;
    ParallelMatrixAlgebra(ParallelMatrixAlgebra&&) = default;
    ParallelMatrixAlgebra& operator=(ParallelMatrixAlgebra&&) = default;

    ParallelMatrixAlgebra(const std::array<Dune::FieldMatrix<scalar_type, P, P>, Q>& m)
      : value_(m)
    {}

    ParallelMatrixAlgebra& operator=(const std::array<Dune::FieldMatrix<scalar_type, P, P>, Q>& m){
      value_ = m;
      return *this;
    }

    static ParallelMatrixAlgebra<X, P> identity(scalar_type multiplier = 1.0){
      std::array<FieldMatrix<scalar_type, P, P>,Q> id;
      std::fill(id.begin(), id.end(), 0.0);
      for(size_t q=0;q<Q;++q)
        for(size_t i=0;i<P;++i)
          id[q][i][i] = multiplier;
      return id;
    }

    // Interface
    void invert() {
      for(size_t i=0;i<Q;++i)
        value_[i].invert();
    }

    void transpose() {
      for(size_t q=0;q<Q;++q)
        for(size_t i=0;i<P;++i){
          for(size_t j=i+1;j<P;++j){
            std::swap(value_[q][i][j], value_[q][j][i]);
          }
        }
    }

    void axpy(scalar_type alpha, X& x, const X& y) const {
      axpy(alpha, x, y, PriorityTag<10>{});
    }

    void mv(X& x) const {
      mv(x, PriorityTag<10>{});
    }

    static ParallelMatrixAlgebra dot(const X& x, const X& y){
      return dot(x, y, PriorityTag<10>{});
    }

    void scale(scalar_type alpha) {
      for(size_t q=0;q<Q;++q)
        value_[q] *= alpha;
    }

    void add(const ParallelMatrixAlgebra& other) {
      for(size_t q=0;q<Q;++q)
        value_[q] += other.value_[q];
    }

    void leftmultiply(const ParallelMatrixAlgebra& other) {
      for(size_t q=0;q<Q;++q)
        value_[q].leftmultiply(other.value_[q]);
    }

    void rightmultiply(const ParallelMatrixAlgebra& other) {
      for(size_t q=0;q<Q;++q)
        value_[q].rightmultiply(other.value_[q]);
    }

    field_type diagonal() const {
      field_type diag = 0.0;
      for(size_t i=0;i<N;++i){
        Simd::lane(i, diag) = value_[i/P][i%P][i%P];
      }
      return diag;
    }

    real_scalar_type frobenius_norm() const {
      using std::sqrt;
      real_scalar_type sum = 0.0;
      for(size_t q=0;q<Q;++q){
        sum += value_[q].frobenius_norm2();
      }
      return sqrt(sum);
    }

    real_type column_norms() const {
      using std::sqrt;
      using std::abs;
      real_type norms = 0.0;
      for(size_t q=0;q<Q;++q){
        for(size_t p1=0;p1<P;++p1){
          for(size_t p2=0;p2<P;++p2){
            Simd::lane(p2+P*q, norms) += abs(value_[q][p1][p2]*value_[q][p1][p2]);
          }
        }
      }
      return sqrt(norms);
    }

    real_scalar_type cond(bool balance=false) const {
      using std::sqrt;
      using std::abs;
      using std::max;
      real_scalar_type max_cond = 0.0;
      for(size_t q=0;q<Q;++q){
        auto cpy = value_[q];
        if(balance){
          std::array<scalar_type, P> diag;
          for(size_t i=0;i<P;++i){
            diag[i] = scalar_type(1.0)/sqrt(value_[q][i][i]);
          }
          for(size_t i=0;i<P;++i){
            for(size_t j=0;j<P;++j){
              cpy[i][j] *= diag[i]*diag[j];
            }
          }
        }
        FieldVector<scalar_type, P> ev;
        FMatrixHelp::eigenValues(cpy,ev);
        max_cond = max(max_cond, abs(ev[P-1])/abs(ev[0]));
      }
      return max_cond;
    }

    void eliminate(const Simd::Mask<field_type>& mask){
      for(size_t q=0;q<Q;++q){
        for(size_t p=0;p<P;++p){
          if(Simd::lane(q*P+p, mask)){
            value_[q][p] = 0.0;
            for(size_t p1=0;p1<P;++p1){
              value_[q][p1][p] = 0.0;
            }
            value_[q][p][p] = 1.0;
          }
        }
      }
    }

    Simd::Mask<field_type> cholesky() {
      Simd::Mask<field_type> dependent_lanes = field_type(0.0)==field_type(1.0); // false
      for(size_t q=0; q<Q;++q){
        auto dc = cholesky_factorize(value_[q]);
        for(const auto& l : dc){
          Simd::lane(q*P+l,dependent_lanes) = true;
        }
      }
      return dependent_lanes;
    }

    friend ParallelMatrixAlgebra operator+ (const ParallelMatrixAlgebra& m1, const ParallelMatrixAlgebra& m2){
      ParallelMatrixAlgebra m = m1;
      m.add(m2);
      return m;
    }

  private:

    template<class X1 = X>
    std::enable_if_t<LoopSIMDcheck<typename X1::field_type,P>()>
    /*void*/ axpy(scalar_type alpha, X& x, const X& y,
                  PriorityTag<7>) const {
      using inner_simd = typename field_type::value_type;
      constexpr size_t ISN = Simd::lanes<inner_simd>();

      std::array<std::array<std::array<std::array<inner_simd, ISN>, P/ISN>, P/ISN>, Q> alpha_m;
      for(size_t q=0; q<Q; ++q){
        for(size_t p1=0;p1<P/ISN;++p1){
          for(size_t p2=0;p2<P/ISN;++p2){
            for(size_t r1=0; r1<ISN;++r1){
              for(size_t r2=0; r2<ISN;++r2){
                Simd::lane(r2, alpha_m[q][p1][p2][r1]) = alpha * value_[q][p1*ISN+r1][p2*ISN+r2];
              }
            }
          }
        }
      }

      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0; i<=x.size()-LOOPSIMD_TILEROWS; i+=LOOPSIMD_TILEROWS){
        for(size_t j=0; j<J; ++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0;p1<P/ISN;++p1){
              std::array<inner_simd, LOOPSIMD_TILEROWS> tile_x, tile_y;
              for(size_t l=0;l<LOOPSIMD_TILEROWS;++l){
                tile_x[l] = Impl::asVector(x[i+l])[j][q*P/ISN+p1];
              }
              for(size_t p2=0;p2<P/ISN;++p2){
                for(size_t l=0;l<LOOPSIMD_TILEROWS;++l){
                  tile_y[l] = Impl::asVector(y[i+l])[j][q*P/ISN+p2];
                }
                mm(tile_y, alpha_m[q][p2][p1], tile_x);
              }
              for(size_t l=0;l<LOOPSIMD_TILEROWS;++l){
                Impl::asVector(x[i+l])[j][q*P/ISN+p1] = tile_x[l];
              }
            }
          }
        }
      }
      // remainer
      for(size_t i=(x.size()/LOOPSIMD_TILEROWS)*LOOPSIMD_TILEROWS; i<x.size();++i){
        for(size_t j=0; j<J; ++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0;p1<P/ISN;++p1){
              std::array<inner_simd, 1> tile_x, tile_y;
              tile_x[0] = Impl::asVector(x[i])[j][q*P/ISN+p1];
              for(size_t p2=0;p2<P/ISN;++p2){
                tile_y[0] = Impl::asVector(y[i])[j][q*P/ISN+p2];
                mm(tile_y, alpha_m[q][p2][p1], tile_x);
              }
              Impl::asVector(x[i])[j][q*P/ISN+p1] = tile_x[0];
            }
          }
        }
      }
    }

    // specialization for P==1
    template<size_t P1 = P>
    std::enable_if_t<P1 == 1>
    /*void*/ axpy(scalar_type alpha, X& x, const X& y,
                  PriorityTag<9>) const {
      field_type mat = 0.0;
      for(size_t i=0;i<N;++i)
        Simd::lane(i,mat) = alpha*value_[i][i];
      x.axpy(mat, y);
    }

    // P==N
    template<size_t P1=P>
    std::enable_if_t<P1==N>
    /*void*/ axpy(scalar_type alpha, X& x, const X& y,
                  PriorityTag<4>) const {
      std::array<field_type, N> alpha_m;
      for(size_t i=0;i<N;++i){
        for(size_t j=0;j<N;++j){
          Simd::lane(j,alpha_m[i]) = alpha*value_[0][i][j];
        }
      }

      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0;i<x.size();++i){
        auto&& xx = Impl::asVector(x[i]);
        auto&& yy = Impl::asVector(y[i]);
        for(size_t j=0;j<J;++j){
          for(size_t k=0;k<N;++k){
            xx[j] = fma(Simd::lane(k, yy[j]), alpha_m[k], xx[j]);
          }
        }
      }
    }

    void axpy(scalar_type alpha, X& x, const X& y,
              PriorityTag<0>) const {
      // This is a fallback as a last resort. This implementation is not
      // optimal as it accesses the lanes of the SIMD objects directly
      auto alpha_m = value_;
      for(auto& mat : alpha_m)
        mat *= alpha;
      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0;i<x.size();++i){
        auto&& xx = Impl::asVector(x[i]);
        auto&& yy = Impl::asVector(y[i]);
        for(size_t j=0;j<J;++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0;p1<P;++p1){
              for(size_t p2=0;p2<P;++p2){
                Simd::lane(p1 + q*P, xx[j]) += Simd::lane(p2 + q*P, yy[j])*alpha_m[q][p2][p1];
              }
            }
          }
        }
      }
    }

#if HAVE_BLAS
    void axpy(scalar_type alpha,
              X& x, const X& y,
              PriorityTag<DUNE_BLOCKKRYLOV_BLAS_PRIORITY>) const {
      int lda = P;
      int ldb = sizeof(field_type)/sizeof(scalar_type); // don't use just N here, because field_type could be overaligned
      int n = x.dim(), m = P;
      int k = P;
      scalar_type beta = 1.0;
      const scalar_type* b= reinterpret_cast<const scalar_type*>(&y[0]);
      scalar_type* c = reinterpret_cast<scalar_type*>(&x[0]);
      for(size_t q=0;q<Q;++q){
        const scalar_type* a= reinterpret_cast<const scalar_type*>(&(value_[q]));
        BLAS::gemm("N", "N", &m, &n, &k, &alpha, a, &lda,
                   b + q*P, &ldb, &beta, c + q*P, &ldb);
      }
    }

#endif

    // mv (only the simple implementation, otherwise call axpy)
    // P==1
    template<size_t P1 = P>
    std::enable_if_t<P1==1>
    /*void*/ mv(X& x,
                PriorityTag<9>) const {
      field_type mat = 0.0;
      for(size_t i=0;i<N;++i)
        Simd::lane(i,mat) = value_[i][i];
      x *= mat;
    }

    // fall back: call axpy
    void mv(X& x,
            PriorityTag<0>) const {
      X y(x);
      x = 0.0;
      axpy(1.0, x, y, PriorityTag<10>{});
    }

    // dot
    template<class X1 = X>
    static std::enable_if_t<LoopSIMDcheck<typename X1::field_type,P>(), ParallelMatrixAlgebra>
    dot(const X& x, const X& y,
        PriorityTag<7>){
      using inner_simd = typename field_type::value_type;
      constexpr size_t ISN = Simd::lanes<inner_simd>();

      std::array<std::array<std::array<std::array<inner_simd, ISN>, P/ISN>, P/ISN>, Q> result;
      for(size_t q=0;q<Q;++q){
        for(size_t p1=0;p1<P/ISN;++p1){
          for(size_t p2=0;p2<P/ISN;++p2){
            for(size_t r=0;r<ISN;++r){
              result[q][p1][p2][r] = 0.0;
            }
          }
        }
      }

      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0; i<=x.size()-LOOPSIMD_TILEROWS; i+=LOOPSIMD_TILEROWS){
        for(size_t j=0;j<J;++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0; p1<P/ISN;++p1){
              std::array<inner_simd, LOOPSIMD_TILEROWS> tile_x, tile_y;
              for(size_t l=0;l<LOOPSIMD_TILEROWS;++l){
                tile_x[l] = Impl::asVector(x[i+l])[j][q*P/ISN+p1];
              }
              for(size_t p2=0;p2<P/ISN;++p2){
                for(size_t l=0;l<LOOPSIMD_TILEROWS;++l){
                  tile_y[l] = Impl::asVector(y[i+l])[j][q*P/ISN+p2];
                }
                mtm(tile_y, tile_x, result[q][p2][p1]);
              }
            }
          }
        }
      }

      // remainer
      for(size_t i=(x.size()/LOOPSIMD_TILEROWS)*LOOPSIMD_TILEROWS; i<x.size(); ++i){
        for(size_t j=0;j<J;++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0; p1<P/ISN;++p1){
              std::array<inner_simd, 1> tile_x, tile_y;
              tile_x[0] = Impl::asVector(x[i])[j][q*P/ISN+p1];
              for(size_t p2=0;p2<P/ISN;++p2){
                tile_y[0] = Impl::asVector(y[i])[j][q*P/ISN+p2];
                mtm(tile_y, tile_x, result[q][p2][p1]);
              }
            }
          }
        }
      }

      // copy result to the BlockAlgebra
      ParallelMatrixAlgebra r;
      for(size_t q=0;q<Q;++q){
        for(size_t p1=0;p1<P/ISN;++p1){
          for(size_t p2=0;p2<P/ISN;++p2){
            for(size_t r1=0;r1<ISN;++r1){
              for(size_t r2=0; r2<ISN; ++r2){
                r.value_[q][p1*ISN+r1][p2*ISN+r2] = Simd::lane(r2,result[q][p1][p2][r1]);
              }
            }
          }
        }
      }
      return r;
    }

    // P==1
    template<size_t P1 = P>
    static std::enable_if_t<P1 == 1, ParallelMatrixAlgebra>
    /*void*/ dot(const X& x, const X& y,
                 PriorityTag<9>){
      auto prod = x.dot(y);
      ParallelMatrixAlgebra result;
      for(size_t l=0;l<Simd::lanes(prod);++l)
        result.value_[l][0][0] = Simd::lane(l,prod);
      return result;
    }

    // P==N
    template<size_t P1 = P>
    static std::enable_if_t<P1 == N, ParallelMatrixAlgebra>
    /*void*/ dot(const X& x, const X& y,
                 PriorityTag<4>){
      std::array<field_type, N> result = filledArray<N>(field_type(0.0));
      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0;i<x.size();++i){
        auto&& xx = Impl::asVector(x[i]);
        auto&& yy = Impl::asVector(y[i]);
        for(size_t j=0;j<J;++j){
          for(size_t k=0;k<N;++k){
            result[k] = fma(Simd::lane(k, xx[j]), yy[j], result[k]);
          }
        }
      }
      ParallelMatrixAlgebra r;
      for(size_t i=0;i<N;++i){
        for(size_t j=0;j<N;++j){
          r.value_[0][i][j] = Simd::lane(i, result[j]);
        }
      }
      return r;
    }

    // fall back
    static ParallelMatrixAlgebra dot(const X& x, const X& y,
                                     PriorityTag<0>){
      ParallelMatrixAlgebra result;
      size_t J = Impl::asVector(x[0]).size();
      for(size_t i=0;i<x.size();++i){
        auto&& xx = Impl::asVector(x[i]);
        auto&& yy = Impl::asVector(y[i]);
        for(size_t j=0;j<J;++j){
          for(size_t q=0;q<Q;++q){
            for(size_t p1=0;p1<P;++p1){
              for(size_t p2=0;p2<P;++p2){
                result.value_[q][p2][p1] += Simd::lane(p1+q*P, xx[j])*Simd::lane(p2+q*P, yy[j]);
              }
            }
          }
        }
      }
      return result;
    }


#if HAVE_BLAS
    static ParallelMatrixAlgebra dot(const X& x, const X& y,
                                     PriorityTag<DUNE_BLOCKKRYLOV_BLAS_PRIORITY>){ // BLOCKKRYLOV_BLAS_PRIORITY is 8 if set during configuration otherwise 5
      int n = P, m = P;
      int k = x.dim();
      int lda = sizeof(field_type)/sizeof(scalar_type); // don't use just N here, because field_type could be overaligned
      int ldc = P;
      scalar_type alpha = 1.0;
      scalar_type beta= 0.0;
      const scalar_type* a = reinterpret_cast<const scalar_type*>(&x[0]);
      const scalar_type* b = reinterpret_cast<const scalar_type*>(&y[0]);
      ParallelMatrixAlgebra result;
      for(size_t q=0;q<Q;++q){
        scalar_type* c = reinterpret_cast<scalar_type*>(&result.value_[q]);
        BLAS::gemm("N", "T", &m, &n, &k, &alpha, a+q*P, &lda,
                   b+q*P, &lda, &beta, c, &ldc);
      }
      return result;
    }

#endif
  };
  /** @} end documentation */
}

#endif
