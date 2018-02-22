// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_ASYNCBLOCKPRECONDITIONER_HH
#define DUNE_ISTL_ASYNCBLOCKPRECONDITIONER_HH

#include <dune/common/parallel/distributeddataexchange.hh>

namespace Dune {
  template<class X, class Y, class T=Preconditioner<X,Y> >
  class AsyncBlockPreconditioner : public Preconditioner<X, Y> {
  public:
    typedef X domain_type;
    typedef Y range_type;
    typedef typename X::field_type field_type;

    AsyncBlockPreconditioner(T& p, AsyncDistributedDataExchange<X>& u)
      : preconditioner_(p)
      , updater_(u)
    {}

    virtual void pre(X& x, Y& b)
    {
      preconditioner_.pre(x, b);
      updater_.sendUpdate(b);
      updater_.project(b);
      updater_.addAll(b);
    }

    virtual void apply (X& v, const Y& d)
    {
      preconditioner_.apply(v, d);
      updater_.sendUpdate(v);
      updater_.project(v);
      updater_.addReady(v);
    }

    virtual void post(X& x)
    {
      preconditioner_.post(x);
    }

    virtual SolverCategory::Category category() const
    {
      return SolverCategory::overlapping;
    }

  private:
    T& preconditioner_;
    AsyncDistributedDataExchange<X>& updater_;
  };
}
#endif
