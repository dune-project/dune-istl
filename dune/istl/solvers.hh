// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_SOLVERS_HH
#define DUNE_ISTL_SOLVERS_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <type_traits>

#include "istlexception.hh"
#include "operators.hh"
#include "scalarproducts.hh"
#include "solver.hh"
#include "preconditioner.hh"
#include <dune/common/deprecated.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/conditional.hh>
#include <dune/common/rangeutilities.hh>
#include <dune/common/timer.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/typetraits.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/eigenvalue/arpackpp.hh>

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

    //! \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      // clear solver statistics
      res.clear();

      // start a timer
      Timer watch;

      // prepare preconditioner
      _prec->pre(x,b);

      // overwrite b with defect
      _op->applyscaleadd(-1,x,b);

      // compute norm, \todo parallelization
      real_type def0 = _sp->norm(b);

      // printing
      if (_verbose>0)
      {
        std::cout << "=== LoopSolver" << std::endl;
        if (_verbose>1)
        {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,def0);
        }
      }

      // allocate correction vector
      X v(x);

      // iteration loop
      int i=1; real_type def=def0;
      for ( ; i<=_maxit; i++ )
      {
        v = 0;                      // clear correction
        _prec->apply(v,b);           // apply preconditioner
        x += v;                     // update solution
        _op->applyscaleadd(-1,v,b);  // update defect
        real_type defnew=_sp->norm(b);  // comp defect norm
        if (_verbose>1)             // print
          this->printOutput(std::cout,i,defnew,def);
        //std::cout << i << " " << defnew << " " << defnew/def << std::endl;
        def = defnew;               // update norm
        if (all_true(def<def0*_reduction) || max_value(def)<1E-30)    // convergence check
        {
          res.converged  = true;
          break;
        }
      }

      //correct i which is wrong if convergence was not achieved.
      i=std::min(_maxit,i);

      // print
      if (_verbose==1)
        this->printOutput(std::cout,i,def);

      // postprocess preconditioner
      _prec->post(x);

      // fill statistics
      res.iterations = i;
      res.reduction = max_value(def/def0);
      res.conv_rate  = pow(res.reduction,1.0/i);
      res.elapsed = watch.elapsed();

      // final print
      if (_verbose>0)
      {
        std::cout << "=== rate=" << res.conv_rate
                  << ", T=" << res.elapsed
                  << ", TIT=" << res.elapsed/i
                  << ", IT=" << i << std::endl;
      }
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
  };


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

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      res.clear();                  // clear solver statistics
      Timer watch;                // start a timer
      _prec->pre(x,b);             // prepare preconditioner
      _op->applyscaleadd(-1,x,b);  // overwrite b with defect

      X p(x);                     // create local vectors
      X q(b);

      real_type def0 = _sp->norm(b); // compute norm

      if (_verbose>0)             // printing
      {
        std::cout << "=== GradientSolver" << std::endl;
        if (_verbose>1)
        {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,def0);
        }
      }

      int i=1; real_type def=def0;   // loop variables
      field_type lambda;
      for ( ; i<=_maxit; i++ )
      {
        p = 0;                      // clear correction
        _prec->apply(p,b);           // apply preconditioner
        _op->apply(p,q);             // q=Ap
        lambda = _sp->dot(p,b)/_sp->dot(q,p); // minimization
        x.axpy(lambda,p);           // update solution
        b.axpy(-lambda,q);          // update defect

        real_type defnew=_sp->norm(b); // comp defect norm
        if (_verbose>1)             // print
          this->printOutput(std::cout,i,defnew,def);

        def = defnew;               // update norm
        if (all_true(def<def0*_reduction) || max_value(def)<1E-30)    // convergence check
        {
          res.converged  = true;
          break;
        }
      }

      //correct i which is wrong if convergence was not achieved.
      i=std::min(_maxit,i);

      if (_verbose==1)                // printing for non verbose
        this->printOutput(std::cout,i,def);

      _prec->post(x);                  // postprocess preconditioner
      res.iterations = i;               // fill statistics
      res.reduction = static_cast<double>(max_value(def/def0));
      res.conv_rate  = pow(res.reduction,1.0/i);
      res.elapsed = watch.elapsed();
      if (_verbose>0)                 // final print
        std::cout << "=== rate=" << res.conv_rate
                  << ", T=" << res.elapsed
                  << ", TIT=" << res.elapsed/i
                  << ", IT=" << i << std::endl;
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
  };



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

    /*!
      \brief Constructor to initialize a CG solver.
      \copydetails IterativeSolver::IterativeSolver(LinearOperator<X,Y>&, Preconditioner<X,Y>&, real_type, int, int)
      \param condition_estimate Whether to calculate an estimate of the condition number.
                                The estimate is given in the InverseOperatorResult returned by apply().
    */
    CGSolver (LinearOperator<X,X>& op, Preconditioner<X,X>& prec,
      real_type reduction, int maxit, int verbose, bool condition_estimate) : IterativeSolver<X,X>(op, prec, reduction, maxit, verbose),
      condition_estimate_(condition_estimate)
    {}

    /*!
      \brief Constructor to initialize a CG solver.
      \copydetails IterativeSolver::IterativeSolver(LinearOperator<X,Y>&, ScalarProduct<X>&, Preconditioner<X,Y>&, real_type, int, int)
      \param condition_estimate Whether to calculate an estimate of the condition number.
                                The estimate is given in the InverseOperatorResult returned by apply().
    */
    CGSolver (LinearOperator<X,X>& op, ScalarProduct<X>& sp, Preconditioner<X,X>& prec,
      real_type reduction, int maxit, int verbose, bool condition_estimate) : IterativeSolver<X,X>(op, sp, prec, reduction, maxit, verbose),
      condition_estimate_(condition_estimate)
    {}


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
      using std::isfinite;

      res.clear();                  // clear solver statistics
      Timer watch;                // start a timer
      _prec->pre(x,b);             // prepare preconditioner
      _op->applyscaleadd(-1,x,b);  // overwrite b with defect

      X p(x);              // the search direction
      X q(x);              // a temporary vector

      real_type def0 = _sp->norm(b); // compute norm

      if (!all_true(isfinite(def0))) // check for inf or NaN
      {
        if (_verbose>0)
          std::cout << "=== CGSolver: abort due to infinite or NaN initial defect"
                    << std::endl;
        DUNE_THROW(SolverAbort, "CGSolver: initial defect=" << def0
                   << " is infinite or NaN");
      }

      if (max_value(def0)<1E-30)    // convergence check
      {
        res.converged  = true;
        res.iterations = 0;               // fill statistics
        res.reduction = 0;
        res.conv_rate  = 0;
        res.elapsed=0;
        if (_verbose>0)                 // final print
          std::cout << "=== rate=" << res.conv_rate
                    << ", T=" << res.elapsed << ", TIT=" << res.elapsed
                    << ", IT=0" << std::endl;
        return;
      }

      if (_verbose>0)             // printing
      {
        std::cout << "=== CGSolver" << std::endl;
        if (_verbose>1) {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,def0);
        }
      }

      // Remember lambda and beta values for condition estimate
      std::vector<real_type> lambdas(0);
      std::vector<real_type> betas(0);

      // some local variables
      real_type def=def0;   // loop variables
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
        lambda = rholast/alpha;     // minimization
        if (condition_estimate_)
          lambdas.push_back(std::real(lambda));
        x.axpy(lambda,p);           // update solution
        b.axpy(-lambda,q);          // update defect

        // convergence test
        real_type defnew=_sp->norm(b); // comp defect norm

        if (_verbose>1)             // print
          this->printOutput(std::cout,i,defnew,def);

        def = defnew;               // update norm
        if (!all_true(isfinite(def))) // check for inf or NaN
        {
          if (_verbose>0)
            std::cout << "=== CGSolver: abort due to infinite or NaN defect"
                      << std::endl;
          DUNE_THROW(SolverAbort,
                     "CGSolver: defect=" << def << " is infinite or NaN");
        }

        if (all_true(def<def0*_reduction) || max_value(def)<1E-30)    // convergence check
        {
          res.converged  = true;
          break;
        }

        // determine new search direction
        q = 0;                      // clear correction
        _prec->apply(q,b);           // apply preconditioner
        rho = _sp->dot(q,b);         // orthogonalization
        beta = rho/rholast;         // scaling factor
        if (condition_estimate_)
          betas.push_back(std::real(beta));
        p *= beta;                  // scale old search direction
        p += q;                     // orthogonalization with correction
        rholast = rho;              // remember rho for recurrence
      }

      //correct i which is wrong if convergence was not achieved.
      i=std::min(_maxit,i);

      if (_verbose==1)                // printing for non verbose
        this->printOutput(std::cout,i,def);

      _prec->post(x);                  // postprocess preconditioner
      res.iterations = i;               // fill statistics
      res.reduction = static_cast<double>(max_value(max_value(def/def0)));
      res.conv_rate  = pow(res.reduction,1.0/i);
      res.elapsed = watch.elapsed();

      if (_verbose>0)                 // final print
      {
        std::cout << "=== rate=" << res.conv_rate
                  << ", T=" << res.elapsed
                  << ", TIT=" << res.elapsed/i
                  << ", IT=" << i << std::endl;
      }


      if (condition_estimate_) {
#if HAVE_ARPACKPP

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
            T[row][row-1] = std::sqrt(betas[row-1]) / lambdas[row-1];
          }

          T[row][row] = 1.0 / lambdas[row];
          if (row > 0) {
            T[row][row] += betas[row-1] / lambdas[row-1];
          }

          if (row < i - 1) {
            T[row][row+1] = std::sqrt(betas[row]) / lambdas[row];
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

        if (_verbose > 0) {
          std::cout << "Min eigv estimate: " << min_eigv << std::endl;
          std::cout << "Max eigv estimate: " << max_eigv << std::endl;
          std::cout << "Condition estimate: " << max_eigv / min_eigv << std::endl;
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
  };


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

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the BiCGSTABSolver aborts when it detects a breakdown.
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::abs;
      const real_type EPSILON=1e-80;
      using std::abs;
      double it;
      field_type rho, rho_new, alpha, beta, h, omega;
      real_type norm, norm_old, norm_0;

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
      res.clear();                // clear solver statistics
      Timer watch;                // start a timer
      _prec->pre(x,r);             // prepare preconditioner
      _op->applyscaleadd(-1,x,r);  // overwrite b with defect

      rt=r;

      norm = norm_old = norm_0 = _sp->norm(r);

      p=0;
      v=0;

      rho   = 1;
      alpha = 1;
      omega = 1;

      if (_verbose>0)             // printing
      {
        std::cout << "=== BiCGSTABSolver" << std::endl;
        if (_verbose>1)
        {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,norm_0);
          //std::cout << " Iter       Defect         Rate" << std::endl;
          //std::cout << "    0" << std::setw(14) << norm_0 << std::endl;
        }
      }

      if ( all_true(norm < (_reduction * norm_0))  || max_value(norm)<1E-30)
      {
        res.converged = 1;
        _prec->post(x);                  // postprocess preconditioner
        res.iterations = 0;             // fill statistics
        res.reduction = 0;
        res.conv_rate  = 0;
        res.elapsed = watch.elapsed();
        return;
      }

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
        if (all_true(abs(rho) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - rho "
                     << rho << " <= EPSILON " << max_value(EPSILON)
                     << " after " << it << " iterations");
        if (all_true(abs(omega) <= EPSILON))
          DUNE_THROW(SolverAbort,"breakdown in BiCGSTAB - omega "
                     << omega << " <= EPSILON " << max_value(EPSILON)
                     << " after " << it << " iterations");


        if (it<1)
          p = r;
        else
        {
          beta = ( rho_new / rho ) * ( alpha / omega );
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

        if ( all_true(abs(h) < EPSILON) )
          DUNE_THROW(SolverAbort,"abs(h) < EPSILON in BiCGSTAB - abs(h) "
                     << abs(h) << " < EPSILON " << max_value(EPSILON)
                     << " after " << it << " iterations");

        alpha = rho_new / h;

        // apply first correction to x
        // x <- x + alpha y
        x.axpy(alpha,y);

        // r = r - alpha*v
        r.axpy(-alpha,v);

        //
        // test stop criteria
        //

        norm = _sp->norm(r);

        if (_verbose>1) // print
        {
          this->printOutput(std::cout,it,norm,norm_old);
        }

        if ( all_true(norm < (_reduction * norm_0)) )
        {
          res.converged = 1;
          break;
        }
        it+=.5;

        norm_old = norm;

        // y = W^-1 * r
        y = 0;
        _prec->apply(y,r);

        // t = A * y
        _op->apply(y,t);

        // omega = < t, r > / < t, t >
        omega = _sp->dot(t,r)/_sp->dot(t,t);

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

        if (_verbose > 1)             // print
        {
          this->printOutput(std::cout,it,norm,norm_old);
        }

        if ( all_true(norm < (_reduction * norm_0))  || max_value(norm)<1E-30)
        {
          res.converged = 1;
          break;
        }

        norm_old = norm;
      } // end for

      //correct i which is wrong if convergence was not achieved.
      it=std::min(static_cast<double>(_maxit),it);

      if (_verbose==1)                // printing for non verbose
        this->printOutput(std::cout,it,norm);

      _prec->post(x);                  // postprocess preconditioner
      res.iterations = static_cast<int>(std::ceil(it));              // fill statistics
      res.reduction = static_cast<double>(max_value(norm/norm_0));
      res.conv_rate  = pow(res.reduction,1.0/it);
      res.elapsed = watch.elapsed();
      if (_verbose>0)                 // final print
        std::cout << "=== rate=" << res.conv_rate
                  << ", T=" << res.elapsed
                  << ", TIT=" << res.elapsed/it
                  << ", IT=" << it << std::endl;
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
  };

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

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      using std::sqrt;
      using std::abs;
      // clear solver statistics
      res.clear();
      // start a timer
      Dune::Timer watch;
      watch.reset();
      // prepare preconditioner
      _prec->pre(x,b);
      // overwrite rhs with defect
      _op->applyscaleadd(-1,x,b);

      // compute residual norm
      real_type def0 = _sp->norm(b);

      // printing
      if(_verbose > 0) {
        std::cout << "=== MINRESSolver" << std::endl;
        if(_verbose > 1) {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,def0);
        }
      }

      // check for convergence
      if( max_value(def0) < 1e-30 ) {
        res.converged = true;
        res.iterations = 0;
        res.reduction = 0;
        res.conv_rate = 0;
        res.elapsed = 0.0;
        // final print
        if(_verbose > 0)
          std::cout << "=== rate=" << res.conv_rate
                    << ", T=" << res.elapsed
                    << ", TIT=" << res.elapsed
                    << ", IT=" << res.iterations
                    << std::endl;
        return;
      }

      // the defect norm
      real_type def = def0;
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

      // some temporary vectors
      X z(b), dummy(b);

      // initialize and clear correction
      z = 0.0;
      _prec->apply(z,b);

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
      q[1] *= real_type(1.0)/beta;
      q[2] = 0.0;

      z *= real_type(1.0)/beta;

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

        q[i2] *= real_type(1.0)/beta;
        z *= real_type(1.0)/beta;

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
        real_type defnew = abs(beta0*xi[i%2]);

          if(_verbose > 1)
            this->printOutput(std::cout,i,defnew,def);

          def = defnew;
          if(all_true(def < def0*_reduction)
              || max_value(def) < 1e-30 || i == _maxit ) {
            res.converged = true;
            break;
          }
        } // end for

        if(_verbose == 1)
          this->printOutput(std::cout,i,def);

        // postprocess preconditioner
        _prec->post(x);
        // fill statistics
        res.iterations = i;
        res.reduction = static_cast<double>(max_value(def/def0));
        res.conv_rate = pow(res.reduction,1.0/i);
        res.elapsed = watch.elapsed();

        // final print
        if(_verbose > 0) {
          std::cout << "=== rate=" << res.conv_rate
                    << ", T=" << res.elapsed
                    << ", TIT=" << res.elapsed/i
                    << ", IT=" << i << std::endl;
        }
    }

  private:

    // helper function to extract the real value of a real or complex number
    inline
    real_type to_real(const real_type & v)
    {
      return v;
    }

    inline
    real_type to_real(const std::complex<real_type> & v)
    {
      return v.real();
    }

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
      cs = cond(norm_dy < eps,
        real_type(1.0),
        cond(norm_dx < eps,
          real_type(0.0),
          cond(norm_dy > norm_dx,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*temp,
            // dy and dx are real in exact arithmetic
            // thus dx*dy is real so we can explicitly enforce it
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*to_real(dx*dy)/norm_dx/norm_dy)));
      sn = cond(norm_dy < eps,
        field_type(0.0),
        cond(norm_dx < eps,
          field_type(1.0),
          cond(norm_dy > norm_dx,
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp),
            // dy and dx is real in exact arithmetic
            // so we don't have to conjugate both of them
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*dy/dx)));
    }

  protected:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
  };

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

    /*!
       \brief Set up RestartedGMResSolver solver.

       \copydoc LoopSolver::LoopSolver(L&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    RestartedGMResSolver (LinearOperator<X,Y>& op, Preconditioner<X,Y>& prec, real_type reduction, int restart, int maxit, int verbose) :
      IterativeSolver<X,Y>::IterativeSolver(op,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Set up RestartedGMResSolver solver.

       \copydoc LoopSolver::LoopSolver(L&,S&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    RestartedGMResSolver (LinearOperator<X,Y>& op, ScalarProduct<X>& sp, Preconditioner<X,Y>& prec, real_type reduction, int restart, int maxit, int verbose) :
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
      apply(x,b,max_value(_reduction),res);
    }

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)

       \note Currently, the RestartedGMResSolver aborts when it detects a
             breakdown.
     */
    virtual void apply (X& x, Y& b, double reduction, InverseOperatorResult& res)
    {
      using std::abs;
      const real_type EPSILON = 1e-80;
      const int m = _restart;
      real_type norm, norm_old = 0.0, norm_0;
      int j = 1;
      std::vector<field_type> s(m+1), sn(m);
      std::vector<real_type> cs(m);
      // need copy of rhs if GMRes has to be restarted
      Y b2(b);
      // helper vector
      Y w(b);
      std::vector< std::vector<field_type> > H(m+1,s);
      std::vector<F> v(m+1,b);

      // start timer
      Dune::Timer watch;
      watch.reset();

      // clear solver statistics and set res.converged to false
      res.clear();
      _prec->pre(x,b);

      // calculate defect and overwrite rhs with it
      _op->applyscaleadd(-1.0,x,b); // b -= Ax
      // calculate preconditioned defect
      v[0] = 0.0; _prec->apply(v[0],b); // r = W^-1 b
      norm_0 = _sp->norm(v[0]);
      norm = norm_0;
      norm_old = norm;

      // print header
      if(_verbose > 0)
        {
          std::cout << "=== RestartedGMResSolver" << std::endl;
          if(_verbose > 1) {
            this->printHeader(std::cout);
            this->printOutput(std::cout,0,norm_0);
          }
        }

      if(all_true(norm_0 < EPSILON)) {
        _prec->post(x);
        res.converged = true;
        if(_verbose > 0) // final print
          print_result(res);
      }

      while(j <= _maxit && res.converged != true) {

        int i = 0;
        v[0] *= real_type(1.0)/norm;
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
          if(all_true(abs(H[i+1][i]) < EPSILON))
            DUNE_THROW(SolverAbort,
                       "breakdown in GMRes - |w| == 0.0 after " << j << " iterations");

          // normalize new vector
          v[i+1] = w; v[i+1] *= real_type(1.0)/H[i+1][i];

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

          // print current iteration statistics
          if(_verbose > 1) {
            this->printOutput(std::cout,j,norm,norm_old);
          }

          norm_old = norm;

          // check convergence
          if(all_true(norm < real_type(reduction) * norm_0))
            res.converged = true;

        } // end for

        // calculate update vector
        w = 0.0;
        update(w,i,H,s,v);
        // and current iterate
        x += w;

        // restart GMRes if convergence was not achieved,
        // i.e. linear defect has not reached desired reduction
        // and if j < _maxit
        if( res.converged != true && j <= _maxit ) {

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
          norm_old = norm;
        }

      } //end while

      // postprocess preconditioner
      _prec->post(x);

      // save solver statistics
      res.iterations = j-1; // it has to be j-1!!!
      res.reduction = static_cast<double>(max_value(norm/norm_0));
      res.conv_rate = pow(res.reduction,1.0/(j-1));
      res.elapsed = watch.elapsed();

      if(_verbose>0)
        print_result(res);

    }

  private :

    void print_result(const InverseOperatorResult& res) const {
      int k = res.iterations>0 ? res.iterations : 1;
      std::cout << "=== rate=" << res.conv_rate
                << ", T=" << res.elapsed
                << ", TIT=" << res.elapsed/k
                << ", IT=" << res.iterations
                << std::endl;
    }

    void update(X& w, int i,
                const std::vector<std::vector<field_type> >& H,
                const std::vector<field_type>& s,
                const std::vector<X>& v) {
      // solution vector of the upper triangular system
      std::vector<field_type> y(s);

      // backsolve
      for(int a=i-1; a>=0; a--) {
        field_type rhs(s[a]);
        for(int b=a+1; b<i; b++)
          rhs -= H[a][b]*y[b];
        y[a] = rhs/H[a][a];

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

    // helper function to extract the real value of a real or complex number
    inline
    real_type to_real(const real_type & v)
    {
      return v;
    }

    inline
    real_type to_real(const std::complex<real_type> & v)
    {
      return v.real();
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
      cs = cond(norm_dy < eps,
        real_type(1.0),
        cond(norm_dx < eps,
          real_type(0.0),
          cond(norm_dy > norm_dx,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*temp,
            real_type(1.0)/sqrt(real_type(1.0) + temp*temp)*to_real(dx*conjugate(dy))/norm_dx/norm_dy)));
      sn = cond(norm_dy < eps,
        field_type(0.0),
        cond(norm_dx < eps,
          field_type(1.0),
          cond(norm_dy > norm_dx,
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp),
            field_type(1.0)/sqrt(real_type(1.0) + temp*temp)*conjugate(dy/dx))));
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
    int _restart;
  };


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

    /*!
       \brief Set up nonlinear preconditioned conjugate gradient solver.

       \copydoc LoopSolver::LoopSolver(L&,P&,double,int,int)
       \param restart number of GMRes cycles before restart
     */
    GeneralizedPCGSolver (LinearOperator<X,X>& op, Preconditioner<X,X>& prec, real_type reduction, int maxit, int verbose, int restart = 10) :
      IterativeSolver<X,X>::IterativeSolver(op,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Set up nonlinear preconditioned conjugate gradient solver.

       \copydoc LoopSolver::LoopSolver(L&,S&,P&,double,int,int)
       \param restart When to restart the construction of
       the Krylov search space.
     */
    GeneralizedPCGSolver (LinearOperator<X,X>& op, ScalarProduct<X>& sp, Preconditioner<X,X>& prec, real_type reduction, int maxit, int verbose, int restart = 10) :
      IterativeSolver<X,X>::IterativeSolver(op,sp,prec,reduction,maxit,verbose),
      _restart(restart)
    {}

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)
     */
    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      res.clear();                      // clear solver statistics
      Timer watch;                    // start a timer
      _prec->pre(x,b);                 // prepare preconditioner
      _op->applyscaleadd(-1,x,b);      // overwrite b with defect

      std::vector<std::shared_ptr<X> > p(_restart);
      std::vector<typename X::field_type> pp(_restart);
      X q(x);                  // a temporary vector
      X prec_res(x);           // a temporary vector for preconditioner output

      p[0].reset(new X(x));

      real_type def0 = _sp->norm(b);    // compute norm
      if ( max_value(def0) < 1E-30 )   // convergence check
      {
        res.converged  = true;
        res.iterations = 0;                     // fill statistics
        res.reduction = 0;
        res.conv_rate  = 0;
        res.elapsed=0;
        if (_verbose>0)                       // final print
          std::cout << "=== rate=" << res.conv_rate
                    << ", T=" << res.elapsed << ", TIT=" << res.elapsed
                    << ", IT=0" << std::endl;
        return;
      }

      if (_verbose>0)                 // printing
      {
        std::cout << "=== GeneralizedPCGSolver" << std::endl;
        if (_verbose>1) {
          this->printHeader(std::cout);
          this->printOutput(std::cout,0,def0);
        }
      }
      // some local variables
      real_type def=def0;       // loop variables
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
      real_type defnew=_sp->norm(b);    // comp defect norm
      if (_verbose>1)                 // print
        this->printOutput(std::cout,++i,defnew,def);
      def = defnew;                   // update norm
      if (all_true(def<def0*_reduction) || max_value(def)<1E-30) // convergence check
      {
        res.converged  = true;
        if (_verbose>0)                       // final print
        {
          std::cout << "=== rate=" << res.conv_rate
                    << ", T=" << res.elapsed
                    << ", TIT=" << res.elapsed
                    << ", IT=" << 1 << std::endl;
        }
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
          defnew=_sp->norm(b);        // comp defect norm

          if (_verbose>1)                     // print
            this->printOutput(std::cout,++i,defnew,def);

          def = defnew;                       // update norm
          if (all_true(def<def0*_reduction) || max_value(def)<1E-30) // convergence check
          {
            res.converged  = true;
            break;
          }
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

      // fill statistics
      res.iterations = i;
      res.reduction = max_value(def/def0);
      res.conv_rate  = pow(res.reduction,1.0/i);
      res.elapsed = watch.elapsed();

      if (_verbose>0)                     // final print
      {
        std::cout << "=== rate=" << res.conv_rate
                  << ", T=" << res.elapsed
                  << ", TIT=" << res.elapsed/i
                  << ", IT=" << i+1 << std::endl;
      }
    }

  private:
    using IterativeSolver<X,X>::_op;
    using IterativeSolver<X,X>::_prec;
    using IterativeSolver<X,X>::_sp;
    using IterativeSolver<X,X>::_reduction;
    using IterativeSolver<X,X>::_maxit;
    using IterativeSolver<X,X>::_verbose;
    int _restart;
  };

  /** @} end documentation */

} // end namespace

#endif
