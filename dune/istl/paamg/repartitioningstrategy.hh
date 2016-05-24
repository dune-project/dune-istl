// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PAAMG_REPARTITIONINGSTRATEGY_HH
#define DUNE_ISTL_PAAMG_REPARTITIONINGSTRATEGY_HH

#include <utility>
#include <dune/common/parametertree.hh>
#include <dune/common/parallel/mpihelper.hh>

namespace Dune {
  namespace Amg {

    struct RepartitioningStrategy
    {

      using size_type = std::size_t;
      using result_type = std::pair<bool,size_type>;

#if HAVE_MPI
      using CollectiveCommunication = Dune::CollectiveCommunication<MPI_Comm>;
#else
      using CollectiveCommunication = Dune::CollectiveCommunication<No_Comm>;
#endif

      virtual result_type operator()(
        const CollectiveCommunication& comm,
        size_type level,
        size_type last_accumulated_level,
        size_type global_unknowns,
        size_type local_unknowns,
        size_type local_nonzeros
        ) const = 0;

      virtual ~RepartitioningStrategy();

    };


    struct DefaultRepartitioningStrategy
      : public RepartitioningStrategy
    {

      using RepartitioningStrategy::size_type;
      using RepartitioningStrategy::result_type;
      using RepartitioningStrategy::CollectiveCommunication;

      result_type operator()(
        const CollectiveCommunication& comm,
        size_type level,
        size_type last_accumulated_level,
        size_type global_unknowns,
        size_type local_unknowns,
        size_type local_nonzeros
        ) const override
      {
        size_type result = static_cast<size_type>(std::ceil(double(global_unknowns)/(_minAggregateSize*_coarsenTarget)));
        return { result < static_cast<size_type>(comm.size()), result };
      }

      template<typename Criterion>
      DefaultRepartitioningStrategy(const Criterion& criterion)
        : _minAggregateSize(criterion.minAggregateSize())
        , _coarsenTarget(criterion.coarsenTarget())
      {}

    private:

      size_type _minAggregateSize;
      size_type _coarsenTarget;

    };


    struct FixedRepartitioningStrategy
      : public RepartitioningStrategy
    {

      using RepartitioningStrategy::size_type;
      using RepartitioningStrategy::result_type;
      using RepartitioningStrategy::CollectiveCommunication;

      result_type operator()(
        const CollectiveCommunication& comm,
        size_type level,
        size_type last_accumulated_level,
        size_type global_unknowns,
        size_type local_unknowns,
        size_type local_nonzeros
        ) const override
      {
        using std::max;
        if ((level + _repartitioning_shift) % _repartitioning_interval == 0)
          return { true, max(size_type(1),comm.size() / _aggregation_factor) };
        else
          return { false, comm.size() };
      }

      FixedRepartitioningStrategy(const ParameterTree& params)
        : _repartitioning_interval(params.get("repartitioningInterval",1))
        , _aggregation_factor(params.get("aggregationFactor",2))
        , _repartitioning_shift(params.get("repartitioningShift",0))
      {}

    private:

      size_type _repartitioning_interval;
      size_type _aggregation_factor;
      size_type _repartitioning_shift;

    };


    struct BandedRepartitioningStrategy
      : public RepartitioningStrategy
    {

      using RepartitioningStrategy::size_type;
      using RepartitioningStrategy::result_type;
      using RepartitioningStrategy::CollectiveCommunication;

      result_type operator()(
        const CollectiveCommunication& comm,
        size_type level,
        size_type last_accumulated_level,
        size_type global_unknowns,
        size_type local_unknowns,
        size_type local_nonzeros
        ) const override
      {
        using std::max;
        if (global_unknowns / comm.size() < _min_unknowns_per_process)
          return { true, max(size_type(1),global_unknowns / _max_unknowns_per_process) };
        else
          return { false, comm.size() };
      }

      BandedRepartitioningStrategy(const ParameterTree& params)
        : _min_unknowns_per_process(params.get("minUnknownsPerProcess",10000))
        , _max_unknowns_per_process(params.get("maxUnknownsPerProcess",160000))
      {}

    private:

      size_type _min_unknowns_per_process;
      size_type _max_unknowns_per_process;

    };

  } // namespace Amg
} // namespace Dune


#endif // DUNE_ISTL_PAAMG_REPARTITIONINGSTRATEGY_HH
