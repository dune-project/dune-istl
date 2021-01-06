// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKKRYLOV_BLOCKBICGSTAB_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLOCKBICGSTAB_HH

#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/blockkrylov/blockinnerproduct.hh>
#include <dune/istl/blockkrylov/utils.hh>

namespace Dune{
  /** @addtogroup ISTL_BK
      @{
  */
  template<class X, size_t P>
  class BlockBiCGSTAB : public IterativeSolver<X, X>
  {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;
    using typename IterativeSolver<X,X>::scalar_real_type;
    using scalar_field_type=Simd::Scalar<field_type>;
    static constexpr size_t K = Simd::lanes<field_type>();
    using Algebra = ParallelMatrixAlgebra<X,P>;

    BlockBiCGSTAB(const std::shared_ptr<LinearOperator<X,X>>& op,
                  const std::shared_ptr<BlockInnerProduct<Algebra>>& sp,
                  const std::shared_ptr<Preconditioner<X,X>>& prec,
                  const ParameterTree& config)
      : IterativeSolver<X,X>(op,sp,prec, config)
      , _residual_ortho(config.get("residual_ortho", 0.0))
      , _bip(sp)
    {}

    BlockBiCGSTAB(std::shared_ptr<LinearOperator<X,X> > op,
                  std::shared_ptr<ScalarProduct<X>> sp,
                  std::shared_ptr<Preconditioner<X,X> > prec,
                  const ParameterTree& config)
      : BlockBiCGSTAB(op,
                      std::dynamic_pointer_cast<BlockInnerProduct<Algebra>>(sp),
                      prec,
                      config)
    {}

    BlockBiCGSTAB(std::shared_ptr<LinearOperator<X,X>> op,
               std::shared_ptr<Preconditioner<X,X>> prec,
               const ParameterTree& config)
      : BlockBiCGSTAB(op,
                      createBlockInnerProduct<X>(P, config.sub("inner_product")),
                      prec,
                      config)
    {}


    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::abs;

      Iteration iteration(*this,res);
      _prec->pre(x,b);             // prepare preconditioner
      _op->applyscaleadd(-1,x,b);  // overwrite b with defect

      real_type norm;
      Algebra sigma;
      if(_residual_ortho > 0){
        sigma = _bip->bnormalize(b,b).get();
        norm = sigma.column_norms();
      }else{
        norm = _bip->norm(b);
      }

      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }
      X p(b);
      p=0.0;
      _prec->apply(p, b);

      X rt0(p), v(p), q(p), z(b), t(b), u(b), w(b);
      Algebra rho, alpha, beta, gamma, rho_sigma;

      for(int it = 1; it < _maxit; ++it){
        _op->apply(p, q);
        //auto alpha_future = _bip->bdot(rt0, q);
        auto alpha_future = _bip->bdot(q, rt0);
        // auto rho_future = _bip->bdot(rt0, b);
        auto rho_future = _bip->bdot(b, rt0);
        z=0.0;
        _prec->apply(z,q);
        alpha = alpha_future.get();
        alpha.invert();
        rho = rho_future.get();
        rho.leftmultiply(alpha);
        rho.axpy(-1.0, b, q); // update S
        rho_sigma = rho;
        if(_residual_ortho>0){
          rho_sigma.rightmultiply(sigma);
        }
        auto gamma_future = _bip->bdot(b,b);
        rho_sigma.axpy(1.0, x, p); // update X
        gamma = gamma_future.get();
        auto residual_cond = gamma.cond(true);
        if(_residual_ortho>0){
          gamma.rightmultiply(sigma);
          gamma.transpose();
          gamma.rightmultiply(sigma);
        }
        norm = sqrt(abs(gamma.diagonal()));
        if(iteration.step(it, norm))
          break;
        // ortho
        if(_residual_ortho*residual_cond > 1.0/std::sqrt(std::numeric_limits<scalar_real_type>::epsilon())
           || (isNaN(residual_cond) && _residual_ortho>0)){
          if(_verbose>1)
            std::cout << "=== orthogonalizing residual" << std::endl;
          gamma = _bip->bnormalize(b,b).get();
          sigma.leftmultiply(gamma);
          t = 0.0;
          _prec->apply(t, b);
        }else{
          t = v;
          rho.axpy(-1.0, t, z);
        }
        _op->apply(t,u);
        auto omega1 = frobenius_product(u,b);
        auto omega2 = frobenius_product(u,u);
        //auto beta_future = _bip->bdot(rt0, u);
        auto beta_future = _bip->bdot(u, rt0);
        w = 0.0;
        _prec->apply(w, u);
        scalar_field_type omega = omega1/omega2;
        if(_residual_ortho>0){
          sigma.axpy(omega, x, t);
        }else
          x.axpy(omega, t);
        b.axpy(-omega, u);
        v=t;
        v.axpy(-omega, w);
        p.axpy(-omega, z);
        z=v;
        beta = beta_future.get();
        beta.leftmultiply(alpha);
        beta.axpy(-1.0, z, p);
        std::swap(p,z);
      }

      _prec->post(x);
    }

  protected:

    scalar_field_type frobenius_product(const X& x, const X& y){
      using Simd::lane;
      using Simd::lanes;
      auto dot = _bip->dot(x,y);
      scalar_field_type sum = 0;
      for(size_t i = 0; i < lanes(dot); ++i)
        sum += lane(i, dot);
      return sum;
    }

    double _residual_ortho = 0.0;
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    std::shared_ptr<BlockInnerProduct<Algebra>> _bip;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
    bool _initial_deflation = false;
  };
  /** @} end documentation */
  DUNE_REGISTER_ITERATIVE_SOLVER("blockbicgstab", blockKrylovSolverCreator<Dune::BlockBiCGSTAB>());
}

#endif
