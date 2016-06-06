// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_COMMGRAPHBASEDVECTORREDISTRIBUTE_HH
#define DUNE_ISTL_COMMGRAPHBASEDVECTORREDISTRIBUTE_HH

#include <vector>
#include <utility>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/unused.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/repartition.hh>
#include <dune/istl/paamg/pinfo.hh>

/**
 * @file
 * @brief Functionality for redistributing a parallel vector in accordance
 *        with the earlier redistribution of a sparse matrix.
 * @author Steffen Müthing
 */
namespace Dune
{
  template<typename  T>
  struct RedistributeInformation
  {

    bool isSetup() const
    {
      return false;
    }
    template<class D>
    void redistribute(const D& from, D& to) const
    {
      DUNE_UNUSED_PARAMETER(from);
      DUNE_UNUSED_PARAMETER(to);
    }

    template<class D>
    void redistributeBackward(D& from, const D& to) const
    {
      DUNE_UNUSED_PARAMETER(from);
      DUNE_UNUSED_PARAMETER(to);
    }

    void resetSetup()
    {}

    void setNoRows(std::size_t size)
    {
      DUNE_UNUSED_PARAMETER(size);
    }

    void setNoCopyRows(std::size_t size)
    {
      DUNE_UNUSED_PARAMETER(size);
    }

    void setNoBackwardsCopyRows(std::size_t size)
    {
      DUNE_UNUSED_PARAMETER(size);
    }

    std::size_t getRowSize(std::size_t index) const
    {
      DUNE_UNUSED_PARAMETER(index);
      return -1;
    }

    std::size_t getCopyRowSize(std::size_t index) const
    {
      DUNE_UNUSED_PARAMETER(index);
      return -1;
    }

    std::size_t getBackwardsCopyRowSize(std::size_t index) const
    {
      DUNE_UNUSED_PARAMETER(index);
      return -1;
    }

  };

#if HAVE_MPI
  template<typename  T, typename T1>
  class RedistributeInformation<OwnerOverlapCopyCommunication<T,T1> >
  {

  public:
    using Comm = OwnerOverlapCopyCommunication<T,T1>;
    using size_type = std::size_t;
    using SizeVector = std::vector<size_type>;
    using IntVector = std::vector<int>;

    RedistributeInformation(
      MPI_Comm partition_comm,
      int partition_owner,
      bool owning_partition,
      SizeVector&& condensed_row_to_old_row_map,
      SizeVector&& old_row_to_condensed_row_map,
      IntVector&& condensed_sizes,
      IntVector&& condensed_offsets
      )
      : _partition_comm(partition_comm)
      , _partition_owner(partition_owner)
      , _owning_partition(owning_partition)
      , _condensed_row_to_old_row_map(std::move(condensed_row_to_old_row_map))
      , _old_row_to_condensed_row_map(std::move(old_row_to_condensed_row_map))
      , _condensed_sizes(std::move(condensed_sizes))
      , _condensed_offsets(std::move(condensed_offsets))
    {}

    RedistributeInformation()
      : _partition_comm(MPI_COMM_NULL)
      , _partition_owner(-1)
      , _owning_partition(false)
    {}

    friend void swap(RedistributeInformation& a, RedistributeInformation& b)
    {
      using std::swap;
      swap(a._partition_comm,b._partition_comm);
      swap(a._partition_owner,b._partition_owner);
      swap(a._owning_partition,b._owning_partition);
      swap(a._condensed_row_to_old_row_map,b._condensed_row_to_old_row_map);
      swap(a._old_row_to_condensed_row_map,b._old_row_to_condensed_row_map);
      swap(a._condensed_sizes,b._condensed_sizes);
      swap(a._condensed_offsets,b._condensed_offsets);
    }

    // We have to allow for copying this thing around, but only as long as it has not been
    // set up. This is important as we assume ownership of the MPI_Comm object and release
    // it in our destructor - and we don't want to do that more than once for obvious reasons.
    RedistributeInformation(const RedistributeInformation& r)
      : RedistributeInformation()
    {
      assert(r._partition_comm == MPI_COMM_NULL);
    }

    // move assignment is fine
    RedistributeInformation& operator=(RedistributeInformation&& r)
    {
      swap(*this,r);
      return *this;
    }

    // We take on ownership of the MPI communicator, so we free it when we're done with it
    ~RedistributeInformation()
    {
      if (_partition_comm != MPI_COMM_NULL)
        MPI_Comm_free(&_partition_comm);
    }

    template<typename V>
    void redistribute(const V& distributed, V& aggregated) const
    {
      // redistribution is pretty straightforward: The mapping between the distributed and
      // the aggregated vector entries is set up in such a way that all distributed DOFs that
      // belong a single distributed rank map to a consecutive section of the aggregated vector
      // We can thus use a simple MPI_Gatherv() to shovel all data to the aggregated vector.
      if (not _owning_partition)
        {
          // We need to copy all of our DOFs that are also present in the aggregated vector to a
          // buffer that can be passed to MPI_Gatherv()
          auto buf = V(_condensed_row_to_old_row_map.size());
          for (size_type i = 0 ; i < _condensed_row_to_old_row_map.size(); ++i)
            buf[i] = distributed[_condensed_row_to_old_row_map[i]];

          // and ship off the data
          MPI_Gatherv(
            buf.data(),
            buf.size(),
            MPITraits<typename V::block_type>::getType(),
            nullptr,nullptr,nullptr,MPI_DATATYPE_NULL,
            _partition_owner,
            _partition_comm
            );
        }
      else
        {
          // On the rank that receives the aggregated vector, we directly write our data into the aggregated
          // vector (note the additional _condensed_offsets lookup on the left hand side of the assignment
          for (size_type i = 0 ; i < _condensed_row_to_old_row_map.size(); ++i)
            aggregated[_condensed_offsets[_partition_owner] + i] = distributed[_condensed_row_to_old_row_map[i]];

          // receive remote data
          MPI_Gatherv(
            MPI_IN_PLACE,0,MPI_DATATYPE_NULL,
            aggregated.data(),
            _condensed_sizes.data(),
            _condensed_offsets.data(),
            MPITraits<typename V::block_type>::getType(),
            _partition_owner,
            _partition_comm
            );
        }
    }

    template<typename V>
    void redistributeBackward(V& distributed, const V& aggregated) const
    {
      // for distributing, we take the easy route: We just broadcast the whole vector to
      // all ranks, and each rank then picks out the entries it needs. This creates a
      // little bit of communication overhead, but it avoids a lot of additional work on
      // the aggregated rank: In order to only send the minimum required data, it would have
      // prepare a custom "care package" for each rank and send that via standard point-to-point
      // communication. By using a broadcast, we avoid those computations and can also benefit
      // from the fan-out communication afforded by the single broadcast operation.

      // target ranks of the broadcast need a buffer for the large vector
      auto buf = V(_owning_partition ? 0 : _condensed_offsets.back());

      auto& aggregated_ = _owning_partition ? const_cast<V&>(aggregated) : buf;

      MPI_Bcast(
        aggregated_.data(),
        aggregated_.size(),
        MPITraits<typename V::block_type>::getType(),
        _partition_owner,
        _partition_comm
        );

      // copy data into the distributed vector
      for (size_type i = 0 ; i < _old_row_to_condensed_row_map.size() ; ++i)
        distributed[i] = aggregated_[_old_row_to_condensed_row_map[i]];
    }

    bool isSetup() const
    {
      return _partition_comm != MPI_COMM_NULL;
    }

    void resetSetup()
    {
      RedistributeInformation tmp;
      swap(*this,tmp);
    }

  private:

    //! communicator for the partition of the vector that gets merged to one processor
    MPI_Comm _partition_comm;
    //! the rank within _partition_comm to which the vector gets merged
    int _partition_owner;
    //! flag that says whether we own the current partition (data gets merged to us)
    bool _owning_partition;
    //! A map from the set of entries that the local process contributes to the merged vector
    //! to the positions of those entries in the complete rank-local vector
    SizeVector _condensed_row_to_old_row_map;
    //! The inverse of the above - a map from the entries of the rank-local vector to the
    //! corresponding entries in the *complete* merged vector - required for redistributing
    //! the vector
    SizeVector _old_row_to_condensed_row_map;
    //! a vector with the number of entries that get sent per rank, required for MPI gathering
    IntVector _condensed_sizes;
    //! a vector with the per-rank offsets into the merged vector
    IntVector _condensed_offsets;

  };

#endif // HAVE_MPI

}

#endif // DUNE_ISTL_COMMGRAPHBASEDVECTORREDISTRIBUTE_HH
