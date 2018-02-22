// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ASYNCLOOPSOLVER_HH
#define DUNE_ISTL_ASYNCLOOPSOLVER_HH

#include "operators.hh"
#include "asyncstoppingcriteria.hh"

namespace Dune {
  template<class X>
  class AsyncLoopSolver{
    LinearOperator<X, X>& _op;
    Preconditioner<X, X>& _prec;
    AsyncStoppingCriteria<X>& _crit;
    int _maxit;
    int _verbose;
  public:
    typedef typename X::field_type field_type;
    AsyncLoopSolver(LinearOperator<X, X>& op, Preconditioner<X, X>& prec, AsyncStoppingCriteria<X>& crit,
               int max_it, int verbose)
      : _op(op)
      , _prec(prec)
      , _crit(crit)
      , _maxit(max_it)
      , _verbose(verbose)
    {}

    virtual void apply (X& x, X& b, InverseOperatorResult& res)
    {
      // clear solver statistics
      res.clear();

      // start a timer
      Timer watch;

      // prepare preconditioner
      _prec.pre(x,b);

      // overwrite b with defect
      _op.applyscaleadd(-1,x,b);

      // compute norm, \todo parallelization
      //real_type def0 = _sp->norm(b);

      // printing
      if (_verbose>0)
      {
        std::cout << "=== LoopSolver" << std::endl;
        if (_verbose>1)
        {
          //this->printHeader(std::cout);
          //this->printOutput(std::cout,0,def0);
        }
      }

      // allocate correction vector
      X v(x);

      // iteration loop
      int i=1;
      //real_type def=def0;
      for ( ; i<=_maxit; i++ )
      {
        v = 0;                      // clear correction
        _prec.apply(v,b);           // apply preconditioner
        x += v;                     // update solution
        _op.applyscaleadd(-1,v,b);  // update defect
        if(_crit.check_stop(b, i, watch.elapsed()))
          break;
      }

      //correct i which is wrong if convergence was not achieved.
      i=std::min(_maxit,i);

      // print
      // if (_verbose==1)
      //   this->printOutput(std::cout,i,def);

      // postprocess preconditioner
      _prec.post(x);

      // fill statistics
      res.iterations = i;
      //res.reduction = static_cast<double>(max_value(def/def0));
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
  };

}

#endif
