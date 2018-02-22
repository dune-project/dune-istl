// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ASYNCSTOPPINGCRITERIA_HH
#define DUNE_ISTL_ASYNCSTOPPINGCRITERIA_HH

namespace Dune {
  template<class X>
  class AsyncStoppingCriteria {
  public:
    virtual bool check_stop(const X&, int, double) = 0;
  };

  template<class X, class C>
  class AsyncNormCheck : public AsyncStoppingCriteria<X>
  {
    typedef typename X::field_type field_type;
    CollectiveCommunication<C> cc_;
    Future<field_type> future_;
    field_type reduction_;
    int iteration_;
    int verbose_;
  public:
    AsyncNormCheck(const C& c, field_type reduction, int verbose)
      : cc_(c)
      , reduction_(reduction)
      , verbose_(verbose)
    {
      if(verbose != 0)
        std::cout << "% Rank\tIteration\tTime\tResidual" << std::endl;
    }

    virtual bool check_stop(const X& x, int iter, double time){
      if(future_.valid() && future_.ready()){
        field_type global_norm = std::sqrt(future_.get());
        if ( verbose_ != 0 )
          std::cout << cc_.rank() << "\t "
                    << iteration_ << "\t "
                    << time << "\t"
                    << global_norm << std::endl;
        if(global_norm < reduction_)
          return true;
      }
      if(!future_.valid()){
        field_type n = x.two_norm2();
        iteration_ = iter;
        future_ = cc_.template iallreduce<std::plus<field_type>>(n);
      }
      return false;
    }
  };

}
#endif
