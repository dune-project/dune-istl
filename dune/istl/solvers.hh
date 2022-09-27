// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERS_HH
#define DUNE_ISTL_SOLVERS_HH

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/common/math.hh>
#include <dune/common/simd/io.hh>
#include <dune/common/simd/simd.hh>
#include <dune/common/std/type_traits.hh>
#include <dune/common/timer.hh>

#include <dune/istl/allocator.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/eigenvalue/arpackpp.hh>
#include <dune/istl/istlexception.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverregistry.hh>

namespace Dune {
  /** @addtogroup ISTL_Solvers
      @{
   */

  /** \file

      \brief   Implementations of the inverse operator interface.

      This file provides various preconditioned Krylov methods.
   */

   //=====================================================================
  // Implementation of this interface
  //=====================================================================

  /*!
     \brief Preconditioned loop solver.

     Implements a preconditioned loop.
     Using this class every Preconditioner can be turned
     into a solver. The solver will apply one preconditioner
     step in each iteration loop.
   */
  template<class X>
  class LoopSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    //! \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      Iteration iteration(*this, res);
      _prec->pre(x,b);

      // overwrite b with defect
      _op->applyscaleadd(-1,x,b);

      // compute norm, \todo parallelization
      real_type def = _sp->norm(b);
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }
      // prepare preconditioner

      // allocate correction vector
      X v(x);

      // iteration loop
      int i=1;
      for ( ; i<=_maxit; i++ )
      {
        v = 0;                      // clear correction
        _prec->apply(v,b);           // apply preconditioner
        x += v;                     // update solution
        _op->applyscaleadd(-1,v,b);  // update defect
        def=_sp->norm(b);  // comp defect norm
        if(iteration.step(i, def))
          break;
      }

      // postprocess preconditioner
      _prec->post(x);
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("loopsolver", defaultIterativeSolverCreator<Dune::LoopSolver>());


  // all these solvers are taken from the SUMO library
  //! gradient method
  template<class X>
  class GradientSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      Iteration iteration(*this, res);
      _prec->pre(x,b);             // prepare preconditioner

      _op->applyscaleadd(-1,x,b);  // overwrite b with defec

      real_type def = _sp->norm(b); // compute norm
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }

      X p(x);                     // create local vectors
      X q(b);

      int i=1;   // loop variables
      field_type lambda;
      for ( ; i<=_maxit; i++ )
      {
        p = 0;                      // clear correction
        _prec->apply(p,b);           // apply preconditioner
        _op->apply(p,q);             // q=Ap
        auto alpha = _sp->dot(q,p);
        lambda = Simd::cond(def==field_type(0.),
                            field_type(0.), // no need for minimization if def is already 0
                            _sp->dot(p,b)/alpha); // minimization
        x.axpy(lambda,p);           // update solution
        b.axpy(-lambda,q);          // update defect

        def =_sp->norm(b); // comp defect norm
        if(iteration.step(i, def))
          break;
      }
      // postprocess preconditioner
      _prec->post(x);
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("gradientsolver", defaultIterativeSolverCreator<Dune::GradientSolver>());

  //! \brief conjugate gradient method
  template<class X>
  class CGSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

  private:
    using typename IterativeSolver<X,X>::scalar_real_type;

  protected:

    static constexpr bool enableConditionEstimate = (std::is_same_v<field_type,float> || std::is_same_v<field_type,double>);

  public:

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
      \brief Constructor to initialize a CG solver.
      \copydetails IterativeSolver::IterativeSolver(const LinearOperator<X,Y>&, Preconditioner<X,Y>&, real_type, int, int)
      \param condition_estimate Whether to calculate an estimate of the condition number.
                                The estimate is given in the InverseOperatorResult returned by apply().
                                This is only supported for float and double field types.
    */
    CGSolver (const LinearOperator<X,X>& op, Preconditioner<X,X>& prec,
      scalar_real_type reduction, int maxit, int verbose, bool condition_estimate) : IterativeSolver<X,X>(op, prec, reduction, maxit, verbose),
      condition_estimate_(condition_estimate)
    {
      if (condition_estimate && !enableConditionEstimate) {
        condition_estimate_ = false;
        std::cerr << "WARNING: Condition estimate was disabled. It is only available for double and float field types!" << std::endl;
      }
    }

    /*!
      \brief Constructor to initialize a CG solver.
      \copydetails IterativeSolver::IterativeSolver(const LinearOperator<X,Y>&, const ScalarProduct<X>&, Preconditioner<X,Y>&, real_type, int, int)
      \param condition_estimate Whether to calculate an estimate of the condition number.
                                The estimate is given in the InverseOperatorResult returned by apply().
                                This is only supported for float and double field types.
    */
    CGSolver (const LinearOperator<X,X>& op, const ScalarProduct<X>& sp, Preconditioner<X,X>& prec,
      scalar_real_type reduction, int maxit, int verbose, bool condition_estimate) : IterativeSolver<X,X>(op, sp, prec, reduction, maxit, verbose),
      condition_estimate_(condition_estimate)
    {
      if (condition_estimate && !(std::is_same<field_type,float>::value || std::is_same<field_type,double>::value)) {
        condition_estimate_ = false;
        std::cerr << "WARNING: Condition estimate was disabled. It is only available for double and float field types!" << std::endl;
      }
    }

    /*!
      \brief Constructor to initialize a CG solver.
      \copydetails IterativeSolver::IterativeSolver(std::shared_ptr<const LinearOperator<X,Y>>, std::shared_ptr<ScalarProduct<X>>, std::shared_ptr<Preconditioner<X,Y>>, real_type, int, int)
      \param condition_estimate Whether to calculate an estimate of the condition number.
                                The estimate is given in the InverseOperatorResult returned by apply().
                                This is only supported for float and double field types.
    */
    CGSolver (std::shared_ptr<const LinearOperator<X,X>> op, std::shared_ptr<ScalarProduct<X>> sp,
              std::shared_ptr<Preconditioner<X,X>> prec,
              scalar_real_type reduction, int maxit, int verbose, bool condition_estimate)
      : IterativeSolver<X,X>(op, sp, prec, reduction, maxit, verbose),
      condition_estimate_(condition_estimate)
    {
      if (condition_estimate && !(std::is_same<field_type,float>::value || std::is_same<field_type,double>::value)) {
        condition_estimate_ = false;
        std::cerr << "WARNING: Condition estimate was disabled. It is only available for double and float field types!" << std::endl;
      }
    }

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the CGSolver aborts when a NaN or infinite defect is
             detected.  However, -ffinite-math-only (implied by -ffast-math)
             can inhibit a result from becoming NaN that really should be NaN.
             E.g. numeric_limits<double>::quiet_NaN()*0.0==0.0 with gcc-5.3
             -ffast-math.
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      Iteration iteration(*this,res);
      _prec->pre(x,b);             // prepare preconditioner

      _op->applyscaleadd(-1,x,b);  // overwrite b with defect

      real_type def = _sp->norm(b); // compute norm
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }

      X p(x);              // the search direction
      X q(x);              // a temporary vector

      // Remember lambda and beta values for condition estimate
      std::vector<real_type> lambdas(0);
      std::vector<real_type> betas(0);

      // some local variables
      field_type rho,rholast,lambda,alpha,beta;

      // determine initial search direction
      p = 0;                          // clear correction
      _prec->apply(p,b);               // apply preconditioner
      rholast = _sp->dot(p,b);         // orthogonalization

      // the loop
      int i=1;
      for ( ; i<=_maxit; i++ )
      {
        // minimize in given search direction p
        _op->apply(p,q);             // q=Ap
        alpha = _sp->dot(p,q);       // scalar product
        lambda = Simd::cond(def==field_type(0.), field_type(0.), rholast/alpha);     // minimization
        if constexpr (enableConditionEstimate)
          if (condition_estimate_)
            lambdas.push_back(std::real(lambda));
        x.axpy(lambda,p);           // update solution
        b.axpy(-lambda,q);          // update defect

        // convergence test
        def=_sp->norm(b); // comp defect norm
        if(iteration.step(i, def))
          break;

        // determine new search direction
        q = 0;                      // clear correction
        _prec->apply(q,b);           // apply preconditioner
        rho = _sp->dot(q,b);         // orthogonalization
        beta = Simd::cond(def==field_type(0.), field_type(0.), rho/rholast);         // scaling factor
        if constexpr (enableConditionEstimate)
          if (condition_estimate_)
            betas.push_back(std::real(beta));
        p *= beta;                  // scale old search direction
        p += q;                     // orthogonalization with correction
        rholast = rho;              // remember rho for recurrence
      }

      _prec->post(x);                  // postprocess preconditioner

      if (condition_estimate_) {
#if HAVE_ARPACKPP
        if constexpr (enableConditionEstimate) {
          using std::sqrt;

          // Build T matrix which has extreme eigenvalues approximating
          // those of the original system
          // (see Y. Saad, Iterative methods for sparse linear systems)

          COND_MAT T(i, i, COND_MAT::row_wise);

          for (auto row = T.createbegin(); row != T.createend(); ++row) {
            if (row.index() > 0)
              row.insert(row.index()-1);
            row.insert(row.index());
            if (row.index() < T.N() - 1)
              row.insert(row.index()+1);
          }
          for (int row = 0; row < i; ++row) {
            if (row > 0) {
              T[row][row-1] = sqrt(betas[row-1]) / lambdas[row-1];
            }

            T[row][row] = 1.0 / lambdas[row];
            if (row > 0) {
              T[row][row] += betas[row-1] / lambdas[row-1];
            }

            if (row < i - 1) {
              T[row][row+1] = sqrt(betas[row]) / lambdas[row];
            }
          }

          // Compute largest and smallest eigenvalue of T matrix and return as estimate
          Dune::ArPackPlusPlus_Algorithms<COND_MAT, COND_VEC> arpack(T);

          real_type eps = 0.0;
          COND_VEC eigv;
          real_type min_eigv, max_eigv;
          arpack.computeSymMinMagnitude (eps, eigv, min_eigv);
          arpack.computeSymMaxMagnitude (eps, eigv, max_eigv);

          res.condition_estimate = max_eigv / min_eigv;

          if (this->_verbose > 0) {
            std::cout << "Min eigv estimate: " << Simd::io(min_eigv) << '\n';
            std::cout << "Max eigv estimate: " << Simd::io(max_eigv) << '\n';
            std::cout << "Condition estimate: "
                      << Simd::io(max_eigv / min_eigv) << std::endl;
          }
        }
#else
      std::cerr << "WARNING: Condition estimate was requested. This requires ARPACK, but ARPACK was not found!" << std::endl;
#endif
      }
    }

  private:
    bool condition_estimate_ = false;

    // Matrix and vector types used for condition estimate
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<real_type,1,1> > COND_MAT;
    typedef Dune::BlockVector<Dune::FieldVector<real_type,1> > COND_VEC;

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("cgsolver", defaultIterativeSolverCreator<Dune::CGSolver>());

  // Ronald Kriemanns BiCG-STAB implementation from Sumo
  //! \brief Bi-conjugate Gradient Stabilized (BiCG-STAB)
  template<class X>
  class BiCGSTABSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the BiCGSTABSolver aborts when it detects a breakdown.
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::abs;
      const Simd::Scalar<real_type> EPSILON=1e-80;
      using std::abs;
      double it;
      field_type rho, rho_new, alpha, beta, h, omega;
      real_type norm;

      //
      // get vectors and matrix
      //
      X& r=b;
      X p(x);
      X v(x);
      X t(x);
      X y(x);
      X rt(x);

      //
      // begin iteration
      //

      // r = r - Ax; rt = r
      Iteration<double> iteration(*this,res);
      _prec->pre(x,r);             // prepare preconditioner

      _op->applyscaleadd(-1,x,r);  // overwrite b with defect

      rt=r;

      norm = _sp->norm(r);
      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }
      p=0;
      v=0;

      rho   = 1;
      alpha = 1;
      omega = 1;

      //
      // iteration
      //

      for (it = 0.5; it < _maxit; it+=.5)
      {
        //
        // preprocess, set vecsizes etc.
        //

        // rho_new = < rt , r >
        rho_new = _sp->dot(rt,r);

        // look if breakdown occurred
        if (Simd::allTrue(abs(rho) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - rho "
                     << Simd::io(rho) << " <= EPSILON " << EPSILON
                     << " after " << it << " iterations");
        if (Simd::allTrue(abs(omega) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - omega "
                     << Simd::io(omega) << " <= EPSILON " << EPSILON
                     << " after " << it << " iterations");


        if (it<1)
          p = r;
        else
        {
          beta = Simd::cond(norm==field_type(0.),
                            field_type(0.), // no need for orthogonalization if norm is already 0
                            ( rho_new / rho ) * ( alpha / omega ));
          p.axpy(-omega,v); // p = r + beta (p - omega*v)
          p *= beta;
          p += r;
        }

        // y = W^-1 * p
        y = 0;
        _prec->apply(y,p);           // apply preconditioner

        // v = A * y
        _op->apply(y,v);

        // alpha = rho_new / < rt, v >
        h = _sp->dot(rt,v);

        if ( Simd::allTrue(abs(h) < EPSILON) )
          DUNE_THROW(SolverAbort,"abs(h) < EPSILON in BiCGSTAB - abs(h) "
                     << Simd::io(abs(h)) << " < EPSILON " << EPSILON
                     << " after " << it << " iterations");

        alpha = Simd::cond(norm==field_type(0.),
                           field_type(0.),
                           rho_new / h);

        // apply first correction to x
        // x <- x + alpha y
        x.axpy(alpha,y);

        // r = r - alpha*v
        r.axpy(-alpha,v);

        //
        // test stop criteria
        //

        norm = _sp->norm(r);
        if(iteration.step(it, norm)){
          break;
        }

        it+=.5;

        // y = W^-1 * r
        y = 0;
        _prec->apply(y,r);

        // t = A * y
        _op->apply(y,t);

        // omega = < t, r > / < t, t >
        h = _sp->dot(t,t);
        omega = Simd::cond(norm==field_type(0.),
                           field_type(0.),
                           _sp->dot(t,r)/h);

        // apply second correction to x
        // x <- x + omega y
        x.axpy(omega,y);

        // r = s - omega*t (remember : r = s)
        r.axpy(-omega,t);

        rho = rho_new;

        //
        // test stop criteria
        //

        norm = _sp->norm(r);
        if(iteration.step(it, norm)){
          break;
        }
      } // end for

      _prec->post(x);                  // postprocess preconditioner
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    template<class CountType>
    using Iteration = typename IterativeSolver<X,X>::template Iteration<CountType>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("bicgstabsolver", defaultIterativeSolverCreator<Dune::BiCGSTABSolver>());

  /*! \brief Minimal Residual Method (MINRES)

     Symmetrically Preconditioned MINRES as in A. Greenbaum, 'Iterative Methods for Solving Linear Systems', pp. 121
     Iterative solver for symmetric indefinite operators.
     Note that in order to ensure the (symmetrically) preconditioned system to remain symmetric, the preconditioner has to be spd.
   */
  template<class X>
  class MINRESSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

    // copy base class constructors
    using IterativeSolver<X,X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::sqrt;
      using std::abs;
      Iteration iteration(*this, res);
      // prepare preconditioner
      _prec->pre(x,b);

      // overwrite rhs with defect
      _op->applyscaleadd(-1.0,x,b); // b -= Ax

      // some temporary vectors
      X z(b), dummy(b);
      z = 0.0;

      // calculate preconditioned defect
      _prec->apply(z,b); // r = W^-1 (b - Ax)
      real_type def = _sp->norm(z);
      if (iteration.step(0, def)){
        _prec->post(x);
        return;
      }

      // recurrence coefficients as computed in Lanczos algorithm
      field_type alpha, beta;
        // diagonal entries of givens rotation
      std::array<real_type,2> c{{0.0,0.0}};
        // off-diagonal entries of givens rotation
      std::array<field_type,2> s{{0.0,0.0}};

      // recurrence coefficients (column k of tridiag matrix T_k)
      std::array<field_type,3> T{{0.0,0.0,0.0}};

      // the rhs vector of the min problem
      std::array<field_type,2> xi{{1.0,0.0}};

      // beta is real and positive in exact arithmetic
      // since it is the norm of the basis vectors (in unpreconditioned case)
      beta = sqrt(_sp->dot(b,z));
      field_type beta0 = beta;

      // the search directions
      std::array<X,3> p{{b,b,b}};
      p[0] = 0.0;
      p[1] = 0.0;
      p[2] = 0.0;

      // orthonormal basis vectors (in unpreconditioned case)
      std::array<X,3> q{{b,b,b}};
      q[0] = 0.0;
      q[1] *= Simd::cond(def==field_type(0.),
                         field_type(0.),
                         real_type(1.0)/beta);
      q[2] = 0.0;

      z *= Simd::cond(def==field_type(0.),
                      field_type(0.),
                      real_type(1.0)/beta);

      // the loop
      int i = 1;
      for( ; i<=_maxit; i++) {

        dummy = z;
        int i1 = i%3,
          i0 = (i1+2)%3,
          i2 = (i1+1)%3;

        // symmetrically preconditioned Lanczos algorithm (see Greenbaum p.121)
        _op->apply(z,q[i2]); // q[i2] = Az
        q[i2].axpy(-beta,q[i0]);
        // alpha is real since it is the diagonal entry of the hermitian tridiagonal matrix
        // from the Lanczos Algorithm
        // so the order in the scalar product doesn't matter even for the complex case
        alpha = _sp->dot(z,q[i2]);
        q[i2].axpy(-alpha,q[i1]);

        z = 0.0;
        _prec->apply(z,q[i2]);

        // beta is real and positive in exact arithmetic
        // since it is the norm of the basis vectors (in unpreconditioned case)
        beta = sqrt(_sp->dot(q[i2],z));

        q[i2] *= Simd::cond(def==field_type(0.),
                            field_type(0.),
                            real_type(1.0)/beta);
        z *= Simd::cond(def==field_type(0.),
                        field_type(0.),
                        real_type(1.0)/beta);

        // QR Factorization of recurrence coefficient matrix
        // apply previous givens rotations to last column of T
        T[1] = T[2];
        if(i>2) {
          T[0] = s[i%2]*T[1];
          T[1] = c[i%2]*T[1];
        }
        if(i>1) {
          T[2] = c[(i+1)%2]*alpha - s[(i+1)%2]*T[1];
          T[1] = c[(i+1)%2]*T[1] + s[(i+1)%2]*alpha;
        }
        else
          T[2] = alpha;

        // update QR factorization
        generateGivensRotation(T[2],beta,c[i%2],s[i%2]);
        // to last column of T_k
        T[2] = c[i%2]*T[2] + s[i%2]*beta;
        // and to the rhs xi of the min problem
        xi[i%2] = -s[i%2]*xi[(i+1)%2];
        xi[(i+1)%2] *= c[i%2];

        // compute correction direction
        p[i2] = dummy;
        p[i2].axpy(-T[1],p[i1]);
        p[i2].axpy(-T[0],p[i0]);
        p[i2] *= real_type(1.0)/T[2];

        // apply correction/update solution
        x.axpy(beta0*xi[(i+1)%2],p[i2]);

        // remember beta_old
        T[2] = beta;

        // check for convergence
        // the last entry in the rhs of the min-problem is the residual
        def = abs(beta0*xi[i%2]);
        if(iteration.step(i, def)){
          break;
        }
      } // end for

      // postprocess preconditioner
      _prec->post(x);
    }

  private:

    void generateGivensRotation(field_type &dx, field_type &dy, real_type &cs, field_type &sn)
    {
      using std::sqrt;
      using std::abs;
      using std::max;
      using std::min;
      const real_type eps = 1e-15;
      real_type norm_dx = abs(dx);
      real_type norm_dy = abs(dy);
      real_type norm_max = max(norm_dx, norm_dy);
      real_type norm_min = min(norm_dx, norm_dy);
      real_type temp = norm_min/norm_max;
      // we rewrite the code in a vectorizable fashion
      cs = Simd::cond(norm_dy < eps,
        real_type(1.0),
        Simd::cond(norm_dx < eps,
          real_type(0.0),
          Simd::cond(norm_dy > norm_dx,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*temp,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)
          )));
      sn = Simd::cond(norm_dy < eps,
        field_type(0.0),
        Simd::cond(norm_dx < eps,
          field_type(1.0),
          Simd::cond(norm_dy > norm_dx,
            // dy and dx are real in exact arithmetic
            // thus dx*dy is real so we can explicitly enforce it
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*dx*dy/norm_dx/norm_dy,
            // dy and dx is real in exact arithmetic
            // so we don't have to conjugate both of them
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*dy/dx
          )));
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("minressolver", defaultIterativeSolverCreator<Dune::MINRESSolver>());

  /**
     \brief implements the Generalized Minimal Residual (GMRes) method

     GMRes solves the unsymmetric linear system Ax = b using the
     Generalized Minimal Residual method as described the SIAM Templates
     book (http://www.netlib.org/templates/templates.pdf).

     \tparam X trial vector, vector type of the solution
     \tparam Y test vector, vector type of the RHS
     \tparam F vector type for orthonormal basis of Krylov space

   */

  template<class X, class Y=X, class F = Y>
  class RestartedGMResSolver : public IterativeSolver<X,Y>
  {
  public:
    using typename IterativeSolver<X,Y>::domain_type;
    using typename IterativeSolver<X,Y>::range_type;
    using typename IterativeSolver<X,Y>::field_type;
    using typename IterativeSolver<X,Y>::real_type;

  protected:
    using typename IterativeSolver<X,X>::scalar_real_type;

    //! \brief field_type Allocator retrieved from domain type
    using fAlloc = ReboundAllocatorType<X,field_type>;
    //! \brief real_type Allocator retrieved from domain type
    using rAlloc = ReboundAllocatorType<X,real_type>;

  public:

    /*!
       \brief Set up RestartedGMResSolver solver.

       \copydoc LoopSolver::LoopSolver(const L&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    RestartedGMResSolver (const LinearOperator<X,Y>& op, Preconditioner<X,Y>& prec, scalar_real_type reduction, int restart, int maxit, int verbose) :
      IterativeSolver<X,Y>::IterativeSolver(op,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Set up RestartedGMResSolver solver.

       \copydoc LoopSolver::LoopSolver(const L&, const S&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    RestartedGMResSolver (const LinearOperator<X,Y>& op, const ScalarProduct<X>& sp, Preconditioner<X,Y>& prec, scalar_real_type reduction, int restart, int maxit, int verbose) :
      IterativeSolver<X,Y>::IterativeSolver(op,sp,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Constructor.

       \copydoc IterativeSolver::IterativeSolver(const L&, const S&,P&,const ParameterTree&)

       Additional parameter:
       ParameterTree Key | Meaning
       ------------------|------------
       restart           | number of GMRes cycles before restart

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    RestartedGMResSolver (std::shared_ptr<const LinearOperator<X,Y> > op, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver<X,Y>::IterativeSolver(op,prec,configuration),
      _restart(configuration.get<int>("restart"))
    {}

    RestartedGMResSolver (std::shared_ptr<const LinearOperator<X,Y> > op, std::shared_ptr<const ScalarProduct<X> > sp, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver<X,Y>::IterativeSolver(op,sp,prec,configuration),
      _restart(configuration.get<int>("restart"))
    {}

    /*!
      \brief Set up RestartedGMResSolver solver.

      \copydoc LoopSolver::LoopSolver(std::shared_ptr<const L>,std::shared_ptr<const S>,std::shared_ptr<P>,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    RestartedGMResSolver (std::shared_ptr<const LinearOperator<X,Y>> op,
                          std::shared_ptr<const ScalarProduct<X>> sp,
                          std::shared_ptr<Preconditioner<X,Y>> prec,
                          scalar_real_type reduction, int restart, int maxit, int verbose) :
      IterativeSolver<X,Y>::IterativeSolver(op,sp,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the RestartedGMResSolver aborts when it detects a
             breakdown.
     */
    virtual void apply (X& x, Y& b, InverseOperatorResult& res)
    {
      apply(x,b,Simd::max(_reduction),res);
    }

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)

       \note Currently, the RestartedGMResSolver aborts when it detects a
             breakdown.
     */
    virtual void apply (X& x, Y& b, [[maybe_unused]] double reduction, InverseOperatorResult& res)
    {
      using std::abs;
      const Simd::Scalar<real_type> EPSILON = 1e-80;
      const int m = _restart;
      real_type norm = 0.0;
      int j = 1;
      std::vector<field_type,fAlloc> s(m+1), sn(m);
      std::vector<real_type,rAlloc> cs(m);
      // need copy of rhs if GMRes has to be restarted
      Y b2(b);
      // helper vector
      Y w(b);
      std::vector< std::vector<field_type,fAlloc> > H(m+1,s);
      std::vector<F> v(m+1,b);

      Iteration iteration(*this,res);

      // clear solver statistics and set res.converged to false
      _prec->pre(x,b);

      // calculate defect and overwrite rhs with it
      _op->applyscaleadd(-1.0,x,b); // b -= Ax
      // calculate preconditioned defect
      v[0] = 0.0; _prec->apply(v[0],b); // r = W^-1 b
      norm = _sp->norm(v[0]);
      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }

      while(j <= _maxit && res.converged != true) {

        int i = 0;
        v[0] *= Simd::cond(norm==real_type(0.),
                           real_type(0.),
                           real_type(1.0)/norm);
        s[0] = norm;
        for(i=1; i<m+1; i++)
          s[i] = 0.0;

        for(i=0; i < m && j <= _maxit && res.converged != true; i++, j++) {
          w = 0.0;
          // use v[i+1] as temporary vector
          v[i+1] = 0.0;
          // do Arnoldi algorithm
          _op->apply(v[i],v[i+1]);
          _prec->apply(w,v[i+1]);
          for(int k=0; k<i+1; k++) {
            // notice that _sp->dot(v[k],w) = v[k]\adjoint w
            // so one has to pay attention to the order
            // in the scalar product for the complex case
            // doing the modified Gram-Schmidt algorithm
            H[k][i] = _sp->dot(v[k],w);
            // w -= H[k][i] * v[k]
            w.axpy(-H[k][i],v[k]);
          }
          H[i+1][i] = _sp->norm(w);
          if(Simd::allTrue(abs(H[i+1][i]) < EPSILON))
            DUNE_THROW(SolverAbort,
                       "breakdown in GMRes - |w| == 0.0 after " << j << " iterations");

          // normalize new vector
          v[i+1] = w;
          v[i+1] *= Simd::cond(norm==real_type(0.),
                               field_type(0.),
                               real_type(1.0)/H[i+1][i]);

          // update QR factorization
          for(int k=0; k<i; k++)
            applyPlaneRotation(H[k][i],H[k+1][i],cs[k],sn[k]);

          // compute new givens rotation
          generatePlaneRotation(H[i][i],H[i+1][i],cs[i],sn[i]);
          // finish updating QR factorization
          applyPlaneRotation(H[i][i],H[i+1][i],cs[i],sn[i]);
          applyPlaneRotation(s[i],s[i+1],cs[i],sn[i]);

          // norm of the defect is the last component the vector s
          norm = abs(s[i+1]);

          iteration.step(j, norm);

        } // end for

        // calculate update vector
        w = 0.0;
        update(w,i,H,s,v);
        // and current iterate
        x += w;

        // restart GMRes if convergence was not achieved,
        // i.e. linear defect has not reached desired reduction
        // and if j < _maxit (do not restart on last iteration)
        if( res.converged != true && j < _maxit ) {

          if(_verbose > 0)
            std::cout << "=== GMRes::restart" << std::endl;
          // get saved rhs
          b = b2;
          // calculate new defect
          _op->applyscaleadd(-1.0,x,b); // b -= Ax;
          // calculate preconditioned defect
          v[0] = 0.0;
          _prec->apply(v[0],b);
          norm = _sp->norm(v[0]);
        }

      } //end while

      // postprocess preconditioner
      _prec->post(x);
    }

  protected :

    void update(X& w, int i,
                const std::vector<std::vector<field_type,fAlloc> >& H,
                const std::vector<field_type,fAlloc>& s,
                const std::vector<X>& v) {
      // solution vector of the upper triangular system
      std::vector<field_type,fAlloc> y(s);

      // backsolve
      for(int a=i-1; a>=0; a--) {
        field_type rhs(s[a]);
        for(int b=a+1; b<i; b++)
          rhs -= H[a][b]*y[b];
        y[a] = Simd::cond(rhs==field_type(0.),
                          field_type(0.),
                          rhs/H[a][a]);

        // compute update on the fly
        // w += y[a]*v[a]
        w.axpy(y[a],v[a]);
      }
    }

    template<typename T>
    typename std::enable_if<std::is_same<field_type,real_type>::value,T>::type conjugate(const T& t) {
      return t;
    }

    template<typename T>
    typename std::enable_if<!std::is_same<field_type,real_type>::value,T>::type conjugate(const T& t) {
      using std::conj;
      return conj(t);
    }

    void
    generatePlaneRotation(field_type &dx, field_type &dy, real_type &cs, field_type &sn)
    {
      using std::sqrt;
      using std::abs;
      using std::max;
      using std::min;
      const real_type eps = 1e-15;
      real_type norm_dx = abs(dx);
      real_type norm_dy = abs(dy);
      real_type norm_max = max(norm_dx, norm_dy);
      real_type norm_min = min(norm_dx, norm_dy);
      real_type temp = norm_min/norm_max;
      // we rewrite the code in a vectorizable fashion
      cs = Simd::cond(norm_dy < eps,
        real_type(1.0),
        Simd::cond(norm_dx < eps,
          real_type(0.0),
          Simd::cond(norm_dy > norm_dx,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*temp,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)
          )));
      sn = Simd::cond(norm_dy < eps,
        field_type(0.0),
        Simd::cond(norm_dx < eps,
          field_type(1.0),
          Simd::cond(norm_dy > norm_dx,
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*dx*conjugate(dy)/norm_dx/norm_dy,
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*conjugate(dy/dx)
          )));
    }


    void
    applyPlaneRotation(field_type &dx, field_type &dy, real_type &cs, field_type &sn)
    {
      field_type temp  =  cs * dx + sn * dy;
      dy = -conjugate(sn) * dx + cs * dy;
      dx = temp;
    }

    using IterativeSolver<X,Y>::_op;
    using IterativeSolver<X,Y>::_prec;
    using IterativeSolver<X,Y>::_sp;
    using IterativeSolver<X,Y>::_reduction;
    using IterativeSolver<X,Y>::_maxit;
    using IterativeSolver<X,Y>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
    int _restart;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("restartedgmressolver", defaultIterativeSolverCreator<Dune::RestartedGMResSolver>());

  /**
     \brief implements the Flexible Generalized Minimal Residual (FGMRes) method (right preconditioned)

     FGMRes solves the right-preconditioned unsymmetric linear system Ax = b using the
     Flexible Generalized Minimal Residual method. It is flexible because the preconditioner can change in every iteration,
     which allows to use Krylov solvers without fixed number of iterations as preconditioners. Needs more memory than GMRes.

     \tparam X trial vector, vector type of the solution
     \tparam Y test vector, vector type of the RHS
     \tparam F vector type for orthonormal basis of Krylov space

   */

  template<class X, class Y=X, class F = Y>
  class RestartedFlexibleGMResSolver : public RestartedGMResSolver<X,Y>
  {
  public:
    using typename RestartedGMResSolver<X,Y>::domain_type;
    using typename RestartedGMResSolver<X,Y>::range_type;
    using typename RestartedGMResSolver<X,Y>::field_type;
    using typename RestartedGMResSolver<X,Y>::real_type;

  private:
    using typename RestartedGMResSolver<X,Y>::scalar_real_type;

    //! \brief field_type Allocator retrieved from domain type
    using fAlloc = typename RestartedGMResSolver<X,Y>::fAlloc;
    //! \brief real_type Allocator retrieved from domain type
    using rAlloc = typename RestartedGMResSolver<X,Y>::rAlloc;

  public:
    // copy base class constructors
    using RestartedGMResSolver<X,Y>::RestartedGMResSolver;

    // don't shadow four-argument version of apply defined in the base class
    using RestartedGMResSolver<X,Y>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)

       \note Currently, the RestartedFlexibleGMResSolver aborts when it detects a
             breakdown.
     */
    void apply (X& x, Y& b, [[maybe_unused]] double reduction, InverseOperatorResult& res) override
    {
      using std::abs;
      const Simd::Scalar<real_type> EPSILON = 1e-80;
      const int m = _restart;
      real_type norm = 0.0;
      int i, j = 1, k;
      std::vector<field_type,fAlloc> s(m+1), sn(m);
      std::vector<real_type,rAlloc> cs(m);
      // helper vector
      Y tmp(b);
      std::vector< std::vector<field_type,fAlloc> > H(m+1,s);
      std::vector<F> v(m+1,b);
      std::vector<X> w(m+1,b);

      Iteration iteration(*this,res);
      // setup preconditioner if it does something in pre

      // calculate residual and overwrite a copy of the rhs with it
      _prec->pre(x, b);
      v[0] = b;
      _op->applyscaleadd(-1.0, x, v[0]); // b -= Ax

      norm = _sp->norm(v[0]); // the residual norm
      if(iteration.step(0, norm)){
        _prec->post(x);
        return;
      }

      // start iterations
      res.converged = false;;
      while(j <= _maxit && res.converged != true)
      {
        v[0] *= (1.0 / norm);
        s[0] = norm;
        for(i=1; i<m+1; ++i)
          s[i] = 0.0;

        // inner loop
        for(i=0; i < m && j <= _maxit && res.converged != true; i++, j++)
        {
          w[i] = 0.0;
          // compute wi = M^-1*vi (also called zi)
          _prec->apply(w[i], v[i]);
          // compute vi = A*wi
          // use v[i+1] as temporary vector for w
          _op->apply(w[i], v[i+1]);
          // do Arnoldi algorithm
          for(int kk=0; kk<i+1; kk++)
          {
            // notice that _sp->dot(v[k],v[i+1]) = v[k]\adjoint v[i+1]
            // so one has to pay attention to the order
            // in the scalar product for the complex case
            // doing the modified Gram-Schmidt algorithm
            H[kk][i] = _sp->dot(v[kk],v[i+1]);
            // w -= H[k][i] * v[kk]
            v[i+1].axpy(-H[kk][i], v[kk]);
          }
          H[i+1][i] = _sp->norm(v[i+1]);
          if(Simd::allTrue(abs(H[i+1][i]) < EPSILON))
            DUNE_THROW(SolverAbort, "breakdown in fGMRes - |w| (-> "
                                     << w[i] << ") == 0.0 after "
                                     << j << " iterations");

          // v[i+1] = w*1/H[i+1][i]
          v[i+1] *= real_type(1.0)/H[i+1][i];

          // update QR factorization
          for(k=0; k<i; k++)
            this->applyPlaneRotation(H[k][i],H[k+1][i],cs[k],sn[k]);

          // compute new givens rotation
          this->generatePlaneRotation(H[i][i],H[i+1][i],cs[i],sn[i]);

          // finish updating QR factorization
          this->applyPlaneRotation(H[i][i],H[i+1][i],cs[i],sn[i]);
          this->applyPlaneRotation(s[i],s[i+1],cs[i],sn[i]);

          // norm of the residual is the last component of vector s
          using std::abs;
          norm = abs(s[i+1]);
          iteration.step(j, norm);
        } // end inner for loop

        // calculate update vector
        tmp = 0.0;
        this->update(tmp, i, H, s, w);
        // and update current iterate
        x += tmp;

        // restart fGMRes if convergence was not achieved,
        // i.e. linear residual has not reached desired reduction
        // and if still j < _maxit (do not restart on last iteration)
        if( res.converged != true && j < _maxit)
        {
          if (_verbose > 0)
            std::cout << "=== fGMRes::restart" << std::endl;
          // get rhs
          v[0] = b;
          // calculate new defect
          _op->applyscaleadd(-1.0, x,v[0]); // b -= Ax;
          // calculate preconditioned defect
          norm = _sp->norm(v[0]); // update the residual norm
        }

      } // end outer while loop

      // post-process preconditioner
      _prec->post(x);
    }

private:
    using RestartedGMResSolver<X,Y>::_op;
    using RestartedGMResSolver<X,Y>::_prec;
    using RestartedGMResSolver<X,Y>::_sp;
    using RestartedGMResSolver<X,Y>::_reduction;
    using RestartedGMResSolver<X,Y>::_maxit;
    using RestartedGMResSolver<X,Y>::_verbose;
    using RestartedGMResSolver<X,Y>::_restart;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("restartedflexiblegmressolver", defaultIterativeSolverCreator<Dune::RestartedFlexibleGMResSolver>());

  /**
   * @brief Generalized preconditioned conjugate gradient solver.
   *
   * A preconditioned conjugate gradient that allows
   * the preconditioner to change between iterations.
   *
   * One example for such preconditioner is AMG when used without
   * a direct coarse solver. In this case the number of iterations
   * performed on the coarsest level might change between applications.
   *
   * In contrast to CGSolver the search directions are stored and
   * the orthogonalization is done explicitly.
   */
  template<class X>
  class GeneralizedPCGSolver : public IterativeSolver<X,X>
  {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

  private:
    using typename IterativeSolver<X,X>::scalar_real_type;

    //! \brief field_type Allocator retrieved from domain type
    using fAlloc = ReboundAllocatorType<X,field_type>;

  public:

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;

    /*!
       \brief Set up nonlinear preconditioned conjugate gradient solver.

       \copydoc LoopSolver::LoopSolver(const L&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    GeneralizedPCGSolver (const LinearOperator<X,X>& op, Preconditioner<X,X>& prec, scalar_real_type reduction, int maxit, int verbose, int restart = 10) :
      IterativeSolver<X,X>::IterativeSolver(op,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Set up nonlinear preconditioned conjugate gradient solver.

       \copydoc LoopSolver::LoopSolver(const L&, const S&,P&,double,int,int)
       \param restart When to restart the construction of
       the Krylov search space.
     */
    GeneralizedPCGSolver (const LinearOperator<X,X>& op, const ScalarProduct<X>& sp, Preconditioner<X,X>& prec, scalar_real_type reduction, int maxit, int verbose, int restart = 10) :
      IterativeSolver<X,X>::IterativeSolver(op,sp,prec,reduction,maxit,verbose),
      _restart(restart)
    {}


     /*!
       \brief Constructor.

       \copydoc IterativeSolver::IterativeSolver(const L&,const S&,P&,const ParameterTree&)

       Additional parameter:
       ParameterTree Key | Meaning
       ------------------|------------
       restart           | number of PCG cycles before restart

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    GeneralizedPCGSolver (std::shared_ptr<const LinearOperator<X,X> > op, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver<X,X>::IterativeSolver(op,prec,configuration),
      _restart(configuration.get<int>("restart"))
    {}

    GeneralizedPCGSolver (std::shared_ptr<const LinearOperator<X,X> > op, std::shared_ptr<const ScalarProduct<X> > sp, std::shared_ptr<Preconditioner<X,X> > prec, const ParameterTree& configuration) :
      IterativeSolver<X,X>::IterativeSolver(op,sp,prec,configuration),
      _restart(configuration.get<int>("restart"))
    {}
    /*!
      \brief Set up nonlinear preconditioned conjugate gradient solver.

      \copydoc LoopSolver::LoopSolver(std::shared_ptr<const L>,std::shared_ptr<const S>,std::shared_ptr<P>,double,int,int)
      \param restart When to restart the construction of
      the Krylov search space.
    */
    GeneralizedPCGSolver (std::shared_ptr<const LinearOperator<X,X>> op,
                          std::shared_ptr<const ScalarProduct<X>> sp,
                          std::shared_ptr<Preconditioner<X,X>> prec,
                          scalar_real_type reduction, int maxit, int verbose,
                          int restart = 10) :
      IterativeSolver<X,X>::IterativeSolver(op,sp,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      Iteration iteration(*this, res);
      _prec->pre(x,b);                 // prepare preconditioner
      _op->applyscaleadd(-1,x,b);      // overwrite b with defect

      std::vector<std::shared_ptr<X> > p(_restart);
      std::vector<field_type,fAlloc> pp(_restart);
      X q(x);                  // a temporary vector
      X prec_res(x);           // a temporary vector for preconditioner output

      p[0].reset(new X(x));

      real_type def = _sp->norm(b);    // compute norm
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }
      // some local variables
      field_type rho, lambda;

      int i=0;
      int ii=0;
      // determine initial search direction
      *(p[0]) = 0;                              // clear correction
      _prec->apply(*(p[0]),b);                   // apply preconditioner
      rho = _sp->dot(*(p[0]),b);             // orthogonalization
      _op->apply(*(p[0]),q);                 // q=Ap
      pp[0] = _sp->dot(*(p[0]),q);           // scalar product
      lambda = rho/pp[0];         // minimization
      x.axpy(lambda,*(p[0]));               // update solution
      b.axpy(-lambda,q);              // update defect

      // convergence test
      def=_sp->norm(b);    // comp defect norm
      ++i;
      if(iteration.step(i, def)){
        _prec->post(x);
        return;
      }

      while(i<_maxit) {
        // the loop
        int end=std::min(_restart, _maxit-i+1);
        for (ii=1; ii<end; ++ii )
        {
          //std::cout<<" ii="<<ii<<" i="<<i<<std::endl;
          // compute next conjugate direction
          prec_res = 0;                                  // clear correction
          _prec->apply(prec_res,b);                       // apply preconditioner

          p[ii].reset(new X(prec_res));
          _op->apply(prec_res, q);

          for(int j=0; j<ii; ++j) {
            rho =_sp->dot(q,*(p[j]))/pp[j];
            p[ii]->axpy(-rho, *(p[j]));
          }

          // minimize in given search direction
          _op->apply(*(p[ii]),q);                     // q=Ap
          pp[ii] = _sp->dot(*(p[ii]),q);               // scalar product
          rho = _sp->dot(*(p[ii]),b);                 // orthogonalization
          lambda = rho/pp[ii];             // minimization
          x.axpy(lambda,*(p[ii]));                   // update solution
          b.axpy(-lambda,q);                  // update defect

          // convergence test
          def = _sp->norm(b);        // comp defect norm

          ++i;
          iteration.step(i, def);
        }
        if(res.converged)
          break;
        if(end==_restart) {
          *(p[0])=*(p[_restart-1]);
          pp[0]=pp[_restart-1];
        }
      }

      // postprocess preconditioner
      _prec->post(x);

    }

  private:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
    int _restart;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("generalizedpcgsolver", defaultIterativeSolverCreator<Dune::GeneralizedPCGSolver>());

  /*! \brief Accelerated flexible conjugate gradient method

     Flexible conjugate gradient method as in Y. Notay 'Flexible conjugate Gradients',
     SIAM J. Sci. Comput Vol. 22, No.4, pp. 1444-1460

     This solver discard cyclic all old directions to speed up computing.
     In exact arithmetic it is exactly the same as the GeneralizedPCGSolver,
     but it is much faster, depending on the operator and dimension.
     On the other hand for large mmax it uses noticeably more memory.

 */
  template<class X>
  class RestartedFCGSolver : public IterativeSolver<X,X> {
  public:
    using typename IterativeSolver<X,X>::domain_type;
    using typename IterativeSolver<X,X>::range_type;
    using typename IterativeSolver<X,X>::field_type;
    using typename IterativeSolver<X,X>::real_type;

  private:
    using typename IterativeSolver<X,X>::scalar_real_type;

  public:
    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X,X>::apply;
    /*!
      \brief Constructor to initialize a RestartedFCG solver.
      \copydetails IterativeSolver::IterativeSolver(const LinearOperator<X,Y>&, Preconditioner<X,Y>&, real_type, int, int, int)
      \param mmax is the maximal number of previous vectors which are orthogonalized against the new search direction.
    */
    RestartedFCGSolver (const LinearOperator<X,X>& op, Preconditioner<X,X>& prec,
                        scalar_real_type reduction, int maxit, int verbose, int mmax = 10) : IterativeSolver<X,X>(op, prec, reduction, maxit, verbose), _mmax(mmax)
    {
    }

    /*!
      \brief Constructor to initialize a RestartedFCG solver.
      \copydetails IterativeSolver::IterativeSolver(const LinearOperator<X,Y>&, const ScalarProduct<X>&, Preconditioner<X,Y>&, real_type, int, int,int)
      \param mmax is the maximal number of previous vectors which are orthogonalized against the new search direction.
    */
    RestartedFCGSolver (const LinearOperator<X,X>& op, const ScalarProduct<X>& sp, Preconditioner<X,X>& prec,
                        scalar_real_type reduction, int maxit, int verbose, int mmax = 10) : IterativeSolver<X,X>(op, sp, prec, reduction, maxit, verbose), _mmax(mmax)
    {
    }

    /*!
      \brief Constructor to initialize a RestartedFCG solver.
      \copydetails IterativeSolver::IterativeSolver(std::shared_ptr<const LinearOperator<X,Y>>, std::shared_ptr<const ScalarProduct<X>>, std::shared_ptr<Preconditioner<X,Y>>, real_type, int, int,int)
      \param mmax is the maximal number of previous vectors which are orthogonalized against the new search direction.
    */
    RestartedFCGSolver (std::shared_ptr<const LinearOperator<X,X>> op,
                        std::shared_ptr<const ScalarProduct<X>> sp,
                        std::shared_ptr<Preconditioner<X,X>> prec,
                        scalar_real_type reduction, int maxit, int verbose,
                        int mmax = 10)
      : IterativeSolver<X,X>(op, sp, prec, reduction, maxit, verbose), _mmax(mmax)
    {}

    /*!
       \brief Constructor.

       \copydoc IterativeSolver::IterativeSolver(const L&, const S&,P&,const ParameterTree&)

       Additional parameter:
       ParameterTree Key | Meaning
       ------------------|------------
       mmax              | number of FCG cycles before restart. default=10

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    RestartedFCGSolver (std::shared_ptr<const LinearOperator<X,X>> op,
                        std::shared_ptr<Preconditioner<X,X>> prec,
                        const ParameterTree& config)
      : IterativeSolver<X,X>(op, prec, config), _mmax(config.get("mmax", 10))
    {}

    RestartedFCGSolver (std::shared_ptr<const LinearOperator<X,X>> op,
                        std::shared_ptr<const ScalarProduct<X>> sp,
                        std::shared_ptr<Preconditioner<X,X>> prec,
                        const ParameterTree& config)
      : IterativeSolver<X,X>(op, sp, prec, config), _mmax(config.get("mmax", 10))
    {}

    /*!
     \brief Apply inverse operator.

     \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the RestartedFCGSolver aborts when a NaN or infinite defect is
             detected.  However, -ffinite-math-only (implied by -ffast-math)
             can inhibit a result from becoming NaN that really should be NaN.
             E.g. numeric_limits<double>::quiet_NaN()*0.0==0.0 with gcc-5.3
             -ffast-math.
     */

    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using rAlloc = ReboundAllocatorType<X,field_type>;
      res.clear();
      Iteration iteration(*this,res);
      _prec->pre(x,b);             // prepare preconditioner
      _op->applyscaleadd(-1,x,b); // overwrite b with defect

      //arrays for interim values:
      std::vector<X> d(_mmax+1, x);                      // array for directions
      std::vector<X> Ad(_mmax+1, x);                    // array for Ad[i]
      std::vector<field_type,rAlloc> ddotAd(_mmax+1,0); // array for <d[i],Ad[i]>
      X w(x);

      real_type def = _sp->norm(b); // compute norm
      if(iteration.step(0, def)){
        _prec->post(x);
        return;
      }

      // some local variables
      field_type alpha;

      // the loop
      int i=1;
      int i_bounded=0;
      while(i<=_maxit && !res.converged) {
        for (; i_bounded <= _mmax && i<= _maxit; i_bounded++) {
          d[i_bounded] = 0;                   // reset search direction
          _prec->apply(d[i_bounded], b);     // apply preconditioner
          w = d[i_bounded];                 // copy of current d[i]
          // orthogonalization with previous directions
          orthogonalizations(i_bounded,Ad,w,ddotAd,d);

          //saving interim values for future calculating
          _op->apply(d[i_bounded], Ad[i_bounded]);                    // save Ad[i]
          ddotAd[i_bounded]=_sp->dot(d[i_bounded],Ad[i_bounded]);    // save <d[i],Ad[i]>
          alpha = _sp->dot(d[i_bounded], b)/ddotAd[i_bounded];      // <d[i],b>/<d[i],Ad[i]>

          //update solution and defect
          x.axpy(alpha, d[i_bounded]);
          b.axpy(-alpha, Ad[i_bounded]);

          // convergence test
          def = _sp->norm(b); // comp defect norm

          iteration.step(i, def);
          i++;
        }
        //restart: exchange first and last stored values
        cycle(Ad,d,ddotAd,i_bounded);
      }

      //correct i which is wrong if convergence was not achieved.
      i=std::min(_maxit,i);

      _prec->post(x);                  // postprocess preconditioner
    }

  private:
    //This function is called every iteration to orthogonalize against the last search directions
    virtual void orthogonalizations(const int& i_bounded,const std::vector<X>& Ad, const X& w, const std::vector<field_type,ReboundAllocatorType<X,field_type>>& ddotAd,std::vector<X>& d) {
      // The RestartedFCGSolver uses only values with lower array index;
      for (int k = 0; k < i_bounded; k++) {
        d[i_bounded].axpy(-_sp->dot(Ad[k], w) / ddotAd[k], d[k]); // d[i] -= <<Ad[k],w>/<d[k],Ad[k]>>d[k]
      }
    }

    // This function is called every mmax iterations to handle limited array sizes.
    virtual void cycle(std::vector<X>& Ad,std::vector<X>& d,std::vector<field_type,ReboundAllocatorType<X,field_type> >& ddotAd,int& i_bounded) {
      // Reset loop index and exchange the first and last arrays
      i_bounded = 1;
      std::swap(Ad[0], Ad[_mmax]);
      std::swap(d[0], d[_mmax]);
      std::swap(ddotAd[0], ddotAd[_mmax]);
    }

  protected:
    int _mmax;
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    using Iteration = typename IterativeSolver<X,X>::template Iteration<unsigned int>;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("restartedfcgsolver", defaultIterativeSolverCreator<Dune::RestartedFCGSolver>());

  /*! \brief Complete flexible conjugate gradient method

     This solver is a simple modification of the RestartedFCGSolver and, if possible, uses mmax old directions.
     It uses noticably more memory, but provides more stability for preconditioner changes.

  */
  template<class X>
  class CompleteFCGSolver : public RestartedFCGSolver<X> {
  public:
    using typename RestartedFCGSolver<X>::domain_type;
    using typename RestartedFCGSolver<X>::range_type;
    using typename RestartedFCGSolver<X>::field_type;
    using typename RestartedFCGSolver<X>::real_type;

    // copy base class constructors
    using RestartedFCGSolver<X>::RestartedFCGSolver;

    // don't shadow four-argument version of apply defined in the base class
    using RestartedFCGSolver<X>::apply;

    // just a minor part of the RestartedFCGSolver apply method will be modified
    virtual void apply (X& x, X& b, InverseOperatorResult& res) override {
      // reset limiter of orthogonalization loop
      _k_limit = 0;
      this->RestartedFCGSolver<X>::apply(x,b,res);
    };

  private:
    // This function is called every iteration to orthogonalize against the last search directions.
    virtual void orthogonalizations(const int& i_bounded,const std::vector<X>& Ad, const X& w, const std::vector<field_type,ReboundAllocatorType<X,field_type>>& ddotAd,std::vector<X>& d) override {
      // This FCGSolver uses values with higher array indexes too, if existent.
      for (int k = 0; k < _k_limit; k++) {
        if(i_bounded!=k)
          d[i_bounded].axpy(-_sp->dot(Ad[k], w) / ddotAd[k], d[k]); // d[i] -= <<Ad[k],w>/<d[k],Ad[k]>>d[k]
      }
      // The loop limit increase, if array is not completely filled.
      if(_k_limit<=i_bounded)
        _k_limit++;

    };

    // This function is called every mmax iterations to handle limited array sizes.
    virtual void cycle(std::vector<X>& Ad, [[maybe_unused]] std::vector<X>& d, [[maybe_unused]] std::vector<field_type,ReboundAllocatorType<X,field_type> >& ddotAd,int& i_bounded) override {
      // Only the loop index i_bounded return to 0, if it reached mmax.
      i_bounded = 0;
      // Now all arrays are filled and the loop in void orthogonalizations can use the whole arrays.
      _k_limit = Ad.size();
    };

    int _k_limit = 0;

  protected:
    using RestartedFCGSolver<X>::_mmax;
    using RestartedFCGSolver<X>::_op;
    using RestartedFCGSolver<X>::_prec;
    using RestartedFCGSolver<X>::_sp;
    using RestartedFCGSolver<X>::_reduction;
    using RestartedFCGSolver<X>::_maxit;
    using RestartedFCGSolver<X>::_verbose;
  };
  DUNE_REGISTER_ITERATIVE_SOLVER("completefcgsolver", defaultIterativeSolverCreator<Dune::CompleteFCGSolver>());
  /** @} end documentation */
} // end namespace

#endif
