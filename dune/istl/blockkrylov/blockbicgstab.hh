// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKKRYLOV_BLOCKBICGSTAB_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLOCKBICGSTAB_HH

/** \file
 * \brief Block BiCGSTAB method for solving systems with multiple right-hand sides
 */


#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/blockkrylov/blockinnerproduct.hh>
#include <dune/istl/blockkrylov/utils.hh>

namespace Dune{
  /** @addtogroup ISTL_Blockkrylov
      @{
  */

  /**
     \brief Implements the block BiCGSTAB method (BlockBiCGSTAB).

     BlockBiCGSTAB solves a linear system Ax = b for
     multiple right-hand sides using the block BiCGSTAB method.

     see
     - El Guennouni, A., Jbilou, K., & Sadok, H. (2003). A block version of BiCGSTAB for linear systems with multiple right-hand sides. Electronic Transactions on Numerical Analysis, 16(129-142), 2.
     - Dreier, N. (2020). Hardware-Oriented Krylov Methods for High-Performance Computing. PhD Thesis. Chapter 6.

     \tparam X vector type
     \tparam P block size
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
      , _residual_ortho(config.get("residual_ortho", 1.0))
      , _bip(sp)
    {}

    BlockBiCGSTAB(std::shared_ptr<LinearOperator<X,X> > op,
                  std::shared_ptr<ScalarProduct<X>> sp,
                  std::shared_ptr<Preconditioner<X,X> > prec,
                  const ParameterTree& config)
      : BlockBiCGSTAB(op,
                      dynamic_cast_or_throw<BlockInnerProduct<Algebra>>(sp),
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
      Algebra sigma, rho, alpha, beta, gamma, rho_sigma;
      sigma = _bip->bnormalize(b,b).get();

      if(Simd::anyTrue(sigma.diagonal() == 0.0)){ // breakdown in cholesky factorization is indicated by zero diagonal entries
        if(_verbose > 1)
          std::cout << "=== amending residual by random right-hand sides" << std::endl;
        fillRandom(b, sigma.diagonal()==0.0);
        gamma = _bip->bnormalize(b,b).get();
        sigma.leftmultiply(gamma);
      }

      norm = sigma.column_norms();

      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }
      X p(b);
      p=0.0;
      _prec->apply(p, b);

      X rt0(p), v(p), q(p), z(b), t(b), u(b), w(b);

      for(int it = 1; it < _maxit; ++it){
        _op->apply(p, q);
        auto alpha_future = _bip->bdot(q, rt0);
        auto rho_future = _bip->bdot(b, rt0);
        z=0.0;
        _prec->apply(z,q);
        alpha = alpha_future.get();
        alpha.invert();
        rho = rho_future.get();
        rho.leftmultiply(alpha);
        rho.axpy(-1.0, b, q); // update S
        rho_sigma = rho;
        rho_sigma.rightmultiply(sigma);
        rho_sigma.axpy(1.0, x, p); // update X
        gamma = _bip->bdot(b,b).get();
        auto residual_cond = gamma.cond(true);
        gamma.rightmultiply(sigma);
        gamma.transpose();
        gamma.rightmultiply(sigma);
        // orthonormalization - improves stability
        if(_residual_ortho*residual_cond > 1.0/std::sqrt(std::numeric_limits<scalar_real_type>::epsilon())
           || (isNaN(residual_cond))){
          if(_verbose>1)
            std::cout << "=== orthogonalizing residual" << std::endl;
          gamma = _bip->bnormalize(b,b).get();
          sigma.leftmultiply(gamma);
          if(Simd::anyTrue(gamma.diagonal() == 0.0)){
            if(_verbose > 1)
              std::cout << "=== Restart due to breakdown" << std::endl;
            fillRandom(b, gamma.diagonal() == 0);
            gamma = _bip->bnormalize(b,b).get();
            sigma.leftmultiply(gamma);
            _prec->apply(p, b);
            rt0 = p;
            u = 0.0; // restart
          }
          t = 0.0;
          _prec->apply(t, b);
        }else{
          t = v;
          rho.axpy(-1.0, t, z);
        }
        norm = sqrt(abs(gamma.diagonal()));
        if(iteration.step(it, norm))
          break;
        _op->apply(t,u);
        auto omega1 = frobenius_product(u,b);
        auto omega2 = frobenius_product(u,u);
        auto beta_future = _bip->bdot(u, rt0);
        w = 0.0;
        _prec->apply(w, u);
        scalar_field_type omega = omega1/omega2;
        sigma.axpy(omega, x, t);
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

    double _residual_ortho = 1.0;
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    std::shared_ptr<BlockInnerProduct<Algebra>> _bip;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  /** @} end documentation */
  DUNE_REGISTER_ITERATIVE_SOLVER("blockbicgstab", blockKrylovSolverCreator<Dune::BlockBiCGSTAB>());
}

#endif
