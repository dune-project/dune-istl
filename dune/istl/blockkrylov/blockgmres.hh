// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKKRYLOV_BLOCKGMRES_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLOCKGMRES_HH

/** \file
 * \brief Block GMRes method for solving systems with multiple right-hand sides
 */


#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/blockkrylov/blockinnerproduct.hh>
#include <dune/istl/blockkrylov/utils.hh>

namespace Dune {

#ifndef DOXYGEN
  namespace {
    // computes the givens rotation and applies it to m1 and m2
    // assumes that m2 is upper triangular
    template<class Scalar, int N>
    std::array<FieldMatrix<Scalar, N, N>, 4> constructGivensRotation(FieldMatrix<Scalar, N, N>& m1,
                                                                     FieldMatrix<Scalar, N, N>& m2){
      using std::sqrt;
      using std::exp;
      using std::abs;
      using std::arg;
      using std::copysign;
      using namespace std::complex_literals;

      // reconstruct Q from the householder reflectors
      FieldMatrix<Scalar, N, N>
        q11 = 0.0,
        q12 = 0.0,
        q21 = 0.0,
        q22 = 0.0;
      for(size_t i=0; i<N;++i){
        q11[i][i] = 1.0;
        q22[i][i] = 1.0;
      }

      // Perform Householder QR
      for(size_t i=0; i<N; ++i){
        // compute householder reflector
        Dune::FieldVector<Scalar, N+1> v;
        for(size_t j=i;j<N;++j){
          v[j-i] = m1[j][i];
        }
        for(size_t j=0;j<=i;++j){
          v[N-i+j] = m2[j][i];
        }
        Scalar alpha = v.two_norm();
        if constexpr (Impl::isComplexLike<Scalar>::value)
                       alpha = exp(Scalar(1i)*arg(v[0]))*abs(alpha);
        else
          alpha = copysign(alpha, v[0]);
        v /= v[0] + alpha;
        v[0] = 1.0;
        Scalar tau = -2.0/ v.two_norm2();

        // apply householder reflection on all remaining columns
        for(size_t k=i; k<N; ++k){
          Scalar vx = 0.0;
          for(size_t j=i;j<N;++j){
            vx += v[j-i]*m1[j][k];
          }
          for(size_t j=0;j<i+1;++j){
            vx += v[N-i+j]*m2[j][k];
          }
          vx *= tau;
          for(size_t j=i;j<N;++j){
            m1[j][k] += vx*v[j-i];
          }
          for(size_t j=0;j<i+1;++j){
            m2[j][k] += vx*v[N-i+j];
          }
        }

        // apply householder to construct Q^T
        for(size_t k=0; k<N;++k){ // kth column
          Scalar vx = 0.0;
          for(size_t j=i;j<N;++j){
            vx += v[j-i]*q11[j][k];
          }
          for(size_t j=0;j<i+1;++j){
            vx += v[N-i+j]*q21[j][k];
          }
          vx *= tau;
          for(size_t j=i;j<N;++j){
            q11[j][k] += vx*v[j-i];
          }
          for(size_t j=0;j<i+1;++j){
            q21[j][k] += vx*v[N-i+j];
          }
        }
        for(size_t k=0; k<N;++k){ // kth column
          Scalar vx = 0.0;
          for(size_t j=i;j<N;++j){
            vx += v[j-i]*q12[j][k];
          }
          for(size_t j=0;j<i+1;++j){
            vx += v[N-i+j]*q22[j][k];
          }
          vx *= tau;
          for(size_t j=i;j<N;++j){
            q12[j][k] += vx*v[j-i];
          }
          for(size_t j=0;j<i+1;++j){
            q22[j][k] += vx*v[N-i+j];
          }
        }
      }
      return {q11, q21, q12, q22};
    }
  }

  template<class Algebra>
  class GivensRotation{
    std::array<Algebra,4> _q;

  public:
    GivensRotation(std::array<Algebra, 4> q)
      : _q(q)
    {}

    void operator()(Algebra& m1, Algebra& m2) const{
      Algebra tmp1 = m1, tmp2 = m2;
      m1.leftmultiply(_q[0]);
      tmp2.leftmultiply(_q[2]);
      m1.add(tmp2);
      m2.leftmultiply(_q[3]);
      tmp1.leftmultiply(_q[1]);
      m2.add(tmp1);
    }
  };

  template<class X, size_t P>
  GivensRotation<ParallelMatrixAlgebra<X,P>>
  constructGivensRotation(ParallelMatrixAlgebra<X,P>& h11,
                          ParallelMatrixAlgebra<X,P>& h12)
  {
    using scalar_type = Simd::Scalar<typename X::field_type>;
    constexpr size_t S = Simd::lanes<typename X::field_type>();
    constexpr size_t Q = S/P;
    std::array<std::array<FieldMatrix<scalar_type, P, P>, Q>, 4> ts;
    for(size_t q=0;q<Q;++q){
      auto t = constructGivensRotation(h11.value_[q], h12.value_[q]);
      for(size_t i=0;i<4;++i)
        ts[i][q] = t[i];
    }
    return GivensRotation<ParallelMatrixAlgebra<X,P>>({ts[0], ts[1], ts[2], ts[3]});
  }
#endif

  /** @addtogroup ISTL_Blockkrylov
      @{
  */
  /**
     \brief Implements the block GMRes method (BlockGMRes).

     BlockGMRes solves the linear system Ax = b for multiple right-hand
     sides using the block method.

     See
     - Saad, Y. (2003). Iterative methods for sparse linear systems. Society for Industrial and Applied Mathematics. Section 6.12.
     - Dreier, N. (2020). Hardware-Oriented Krylov Methods for High-Performance Computing. PhD Thesis. Chapter 5.

     \tparam X domain vector type
     \tparam X range vector type
     \tparam P block size
     \tparam RIGHT_PREC whether to use right or left preconditioning
  */

  // TODO: allow different range vector type
  template<class X, class Y=X, size_t P = Simd::lanes<typename X::field_type>(), bool RIGHT_PREC=true>
  class BlockGMRes : public IterativeSolver<X, Y>
  {
  public:
    using typename IterativeSolver<X,Y>::domain_type;
    using typename IterativeSolver<X,Y>::range_type;
    using typename IterativeSolver<X,Y>::field_type;
    using typename IterativeSolver<X,Y>::real_type;
    using typename IterativeSolver<X,Y>::scalar_real_type;

    using V = std::conditional_t<RIGHT_PREC, Y, X>; // Krylov space vector type
    using scalar_field_type=Simd::Scalar<field_type>;
    static constexpr size_t K = Simd::lanes<field_type>();
    typedef FieldMatrix<scalar_field_type, K, K> FieldMatrixBlock;
    using Algebra = ParallelMatrixAlgebra<V, P>;

    BlockGMRes(const std::shared_ptr<LinearOperator<X,Y>>& op,
               const std::shared_ptr<BlockInnerProduct<Algebra>>& sp,
               const std::shared_ptr<Preconditioner<X,Y>>& prec,
               const ParameterTree& config = {})
      : IterativeSolver<X,Y>(op,sp,prec, config)
      , _bip(sp)
      , _restart(config.get("restart", 20))
      , _replace_breakdown(config.get("replace_breakdown", false))
    {}

    BlockGMRes(std::shared_ptr<LinearOperator<X,Y> > op,
               std::shared_ptr<ScalarProduct<V>> sp,
               std::shared_ptr<Preconditioner<X,Y> > prec,
               const ParameterTree& config)
      : BlockGMRes(op,
                   dynamic_cast_or_throw<BlockInnerProduct<Algebra>>(sp),
                   prec,
                   config)
    {}

    BlockGMRes(std::shared_ptr<LinearOperator<X,Y>> op,
            std::shared_ptr<Preconditioner<X,Y>> prec,
            const ParameterTree& config)
      : BlockGMRes(op,
                   createBlockInnerProduct<V>(P, config.sub("inner_product")),
                   prec,
                   config)
    {}

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,Y>::apply;

    virtual void apply (X& x, Y& b, InverseOperatorResult& res)
    {
      constexpr scalar_field_type mach_eps = std::numeric_limits<scalar_field_type>::epsilon();
      Iteration iteration(*this,res);
      _prec->pre(x,b);             // prepare preconditioner

      Y b0(b);
      _op->applyscaleadd(-1.0,x,b);

      // Krylov basis
      std::vector<V> v(_restart+1);
      if constexpr (RIGHT_PREC){
        std::fill(v.begin(), v.end(), b);
      }else{
        std::fill(v.begin(), v.end(), x);
        v[0] = 0.0;
        _prec->apply(v[0], b);
      }

      // orthonormalize v[0]
      // transformed residual
      Algebra s0 = _bip->bnormalize(v[0], v[0]).get();
      std::vector<Algebra> s(_restart+1, s0);

      real_type norm = s[0].column_norms();
      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }

      // transformations
      std::vector<GivensRotation<Algebra>> Q;
      // Hessenberg matrix
      std::vector<std::vector<Algebra>> H(_restart);

      using TMP_type = std::conditional_t<RIGHT_PREC, X, Y>;
      TMP_type tmp;
      if constexpr(RIGHT_PREC){
        tmp = x;
      }else{
        tmp = b;
      }

      int j = 1;
      while(j< _maxit && res.converged != true){
        size_t i = 0;
        for(; i < _restart && !res.converged; ++j){
          // block arnoldi
          if constexpr(RIGHT_PREC){
            tmp = 0.0;
            _prec->apply(tmp, v[i]);
            _op->apply(tmp, v[i+1]);
          }else{
            _op->apply(v[i], tmp);
            v[i+1] = 0.0;
            _prec->apply(v[i+1], tmp);
          }
          H[i].resize(i+1);
          for(size_t k=0;k<i+1; ++k){
            H[i][k] = _bip->bdot(v[i+1],v[k]).get();
            H[i][k].axpy(-1.0, v[i+1], v[k]);
          }
          Algebra normalizer = _bip->bnormalize(v[i+1], v[i+1]).get();

          // handle breakdown
          auto breakdown_mask = normalizer.diagonal() < 100*mach_eps;
          if(Simd::anyTrue(breakdown_mask) && _replace_breakdown){
            std::cout << "=== replacing " << countTrue(breakdown_mask) << " columns and reorthogonalizing" << std::endl;
            fillRandom(v[i+1], breakdown_mask);
            // orthogonalize against previous basis
            for(size_t k=0;k<i+1; ++k){
              auto tmp = _bip->bdot(v[i+1],v[k]).get();
              tmp.rightmultiply(normalizer);
              H[i][k].add(tmp);
            }
            auto tmp = _bip->bnormalize(v[i+1], v[i+1]).get();
            normalizer.leftmultiply(tmp);
          }

          // apply QR transformations
          for(size_t k=0; k<i;++k){
            Q[k](H[i][k], H[i][k+1]);
          }

          // compute new QR transformation
          Q.push_back(constructGivensRotation(H[i][i], normalizer));

          // apply new qr transformation on residual
          s[i+1] = Algebra::identity(0.0);
          Q[i](s[i], s[i+1]);

          i++;
          real_type norm = s[i].column_norms();
          if(iteration.step(j, norm) || j>=_maxit)
            break;
        }
        blockBackSubstitute(H,s,i);

        // compute R^{-1}s*V to get the least-squares solution
        for(size_t k=0;k<i;++k){
          s[k].axpy(1.0, x, v[k]);
        }
        if(!res.converged && j < _maxit ){
          if(_verbose > 0)
            std::cout << "=== BlockGMRes::restart" << std::endl;
          b = b0;
          if constexpr (RIGHT_PREC){
            tmp = 0.0;
            _prec->apply(tmp, x);
            v[0] = b;
            _op->applyscaleadd(-1.0, tmp, v[0]);
          }else{
            _op->applyscaleadd(-1.0, x, b);
            v[0] = 0.0;
            _prec->apply(v[0], b);
          }
          Q.clear();
          s[0] = _bip->bnormalize(v[0], v[0]).get();
        }
      }
      if constexpr (RIGHT_PREC){
        tmp = x;
        x = 0.0;
        _prec->apply(x, tmp);
      }
      _prec->post(x);
    }

  protected:
    void blockBackSubstitute(std::vector<std::vector<Algebra>>& H,
                             std::vector<Algebra>& b,
                             size_t l){
      for(int i=l-1;i>=0; --i){
        b[i].scale(-1.0);
        for(size_t j=i+1;j<l;++j){
          H[j][i].rightmultiply(b[j]);
          b[i].add(H[j][i]);
        }
        b[i].scale(-1.0);
        H[i][i].invert();
        b[i].leftmultiply(H[i][i]);
      }
    }

    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
    std::shared_ptr<BlockInnerProduct<Algebra>> _bip;
    int _restart = 20;
    bool _right_preconditioning = true;
    bool _replace_breakdown = false;
  };
  /** @} end documentation */
  DUNE_REGISTER_ITERATIVE_SOLVER("blockgmres",
                                 [](auto typeList,
                                    const auto& linearOperator,
                                    const auto& scalarProduct,
                                    const auto& preconditioner,
                                    const ParameterTree& config)
                                 {
                                   using Domain = typename Dune::TypeListElement<0, decltype(typeList)>::type;
                                   using Range = typename Dune::TypeListElement<1, decltype(typeList)>::type;
                                   constexpr size_t K = Simd::lanes<typename Domain::field_type>();
                                   std::shared_ptr<InverseOperator<Domain, Range>> solver;
                                   bool right_preconditioning = config.get("right_preconditioning", true);
                                   Hybrid::switchCases(dividersOf<K>(),
                                                       config.get<size_t>("p", K),
                                                       [&](auto pp){
                                                         if(right_preconditioning)
                                                           solver = std::make_shared<BlockGMRes<Domain, Range, (pp.value), true>>(linearOperator, scalarProduct, preconditioner, config);
                                                         else
                                                           solver = std::make_shared<BlockGMRes<Domain, Range, (pp.value), false>>(linearOperator, scalarProduct, preconditioner, config);
                                                       },
                                                       [](auto...){
                                                         DUNE_THROW(Exception, "Invalid parameter P: P must be a divider of the SIMD width");
                                                       });
                                   return solver;
                                 };);
}

#endif
