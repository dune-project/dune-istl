// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_BLOCKCG_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLOCKCG_HH

/** \file
 * \brief Block CG method for solving systems with multiple right-hand sides
 */


#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>

#include <dune/istl/blockkrylov/matrixalgebra.hh>
#include <dune/istl/blockkrylov/blockinnerproduct.hh>
#include <dune/istl/blockkrylov/utils.hh>

namespace Dune{
  /** @addtogroup ISTL_Blockkrylov
      @{
  */

  /**
     \brief Implements the block Conjugate Gradient method (BlockCG).

     BlockCG solves the symmetric positive definite linear system Ax = b for
     multiple right-hand sides using the block Conjugate Gradients.

     See
     - O'Leary, D. P. (1980). The block conjugate gradient algorithm and related methods.
     - Dreier, N. (2020). Hardware-Oriented Krylov Methods for High-Performance Computing. PhD Thesis. Chapter 4.

     \tparam X vector type
     \tparam P block size

  */
  template<class X, std::size_t P = Simd::lanes<typename X::field_type>()>
  class BlockCG
    : public IterativeSolver<X,X>{
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;
    typedef ParallelMatrixAlgebra<X, P> Algebra;

    BlockCG(std::shared_ptr<LinearOperator<X,X> > op,
            std::shared_ptr<BlockInnerProduct<Algebra>> sp,
            std::shared_ptr<Preconditioner<X,X> > prec,
            const ParameterTree& config)
      : IterativeSolver<X,X>(op,sp, prec, config)
      , _bip(sp)
      , _breakdown_restart(config.get("breakdown_restart", Simd::lanes<field_type>()))
    {}

    BlockCG(std::shared_ptr<LinearOperator<X,X> > op,
            std::shared_ptr<ScalarProduct<X>> sp,
            std::shared_ptr<Preconditioner<X,X> > prec,
            const ParameterTree& config)
      : BlockCG(op,
                dynamic_cast_or_throw<BlockInnerProduct<Algebra>>(sp),
                prec,
                config)
    {}

    BlockCG(std::shared_ptr<LinearOperator<X,X>> op,
            std::shared_ptr<Preconditioner<X,X>> prec,
            const ParameterTree& config)
      : BlockCG(op,
                createBlockInnerProduct<X>(P, config.sub("inner_product")),
                prec,
                config)
    {}

  private:
    using typename IterativeSolver<X,X>::scalar_real_type;
    using scalar_field_type = Simd::Scalar<field_type>;
    static constexpr size_t K = Simd::lanes<field_type>();

  public:
    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      Iteration iteration(*this,res);
      _prec->pre(x,b);             // prepare preconditioner

      _op->applyscaleadd(-1,x,b);  // overwrite b with defect

      X p(x);              // the search direction
      X q(x);              // a temporary vector

      // some local variables
      Algebra alpha, gamma, sigma;

      real_type def;
      p = 0.0;                          // clear correction
      _prec->apply(p,b);               // apply preconditioner
      sigma = _bip->bnormalize(b,p).get();

      if(Simd::anyTrue(sigma.diagonal() == 0.0)){ // breakdown in cholesky factorization is indicated by zero diagonal entries
        if(_verbose > 1)
          std::cout << "=== amending residual by random right-hand sides" << std::endl;
        fillRandom(b, sigma.diagonal()==0.0);
        p = 0.0;
        _prec->apply(p,b);               // apply preconditioner
        gamma = _bip->bnormalize(b,p).get();
        sigma.leftmultiply(gamma);
      }

      def = sigma.column_norms();
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }

      Simd::Mask<field_type> breakdown_cols = false;
      size_t breakdown_counter = 0;

      // the loop
      int i=1;
      for ( ; i<=_maxit; i++ )
        {
          // minimize in given search direction p
          _op->apply(p,q);             // q=Ap
          alpha = _bip->bdot(p,q).get();       // inner block product
          alpha.eliminate(breakdown_cols);
          alpha.invert();
          alpha.axpy(-1.0, b, q);         // update defect
          alpha.rightmultiply(sigma);
          alpha.axpy(1.0, x, p);          // update solution
          // determine new search direction
          q = 0;                      // clear correction
          _prec->apply(q,b);           // apply preconditioner

          gamma = _bip->bnormalize(b, q).get(); // orthogonalize residual
          sigma.leftmultiply(gamma);            // update residual transformation

          def = sigma.column_norms();           // compute preconditioned residual norm
          if(iteration.step(i, def))
            break;

          // check for breakdown
          breakdown_cols = gamma.diagonal()==0.0; // breakdown in cholesky factorization is indicated by zero diagonal entries
          breakdown_counter += countTrue(breakdown_cols);
          if(breakdown_counter > _breakdown_restart){
            if(_verbose > 1)
              std::cout << "=== Restart due to breakdown" << std::endl;
            fillRandom(b, breakdown_cols);
            p = 0;
            _prec->apply(p,b);
            gamma = _bip->bnormalize(b, p).get(); // orthogonalize residual
            sigma.leftmultiply(gamma);            // update residual transformation
            breakdown_cols = false;
            breakdown_counter = 0;
          }else{
            gamma.transpose();
            // update P
            gamma.axpy(1.0, q, p);
            std::swap(p,q);
          }
        }

      _prec->post(x);                  // postprocess preconditioner
    }


  protected:

    size_t _breakdown_restart = 0;
    std::shared_ptr<BlockInnerProduct<Algebra>> _bip;
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  /** @} end documentation */
  DUNE_REGISTER_ITERATIVE_SOLVER("blockcg", blockKrylovSolverCreator<Dune::BlockCG>());
}

#endif
