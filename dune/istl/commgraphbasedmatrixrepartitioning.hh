// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_COMMGRAPHBASEDMATRIXREPARTITIONING_HH
#define DUNE_ISTL_COMMGRAPHBASEDMATRIXREPARTITIONING_HH

#include <cassert>
#include <map>
#include <utility>
#include <unordered_map>

#if HAVE_METIS

// Explicitly use C linkage as scotch does not extern "C" in its headers.
// Works because METIS checks whether compiler is C++ and otherwise
// does not use extern "C". Therfore no nested extern "C" will be created
extern "C"
{
#include <metis.h>
}
#endif

#include <dune/common/parametertree.hh>
#include <dune/common/timer.hh>
#include <dune/common/unused.hh>
#include <dune/common/enumutility.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/indicessyncer.hh>
#include <dune/common/parallel/remoteindices.hh>

#include <dune/istl/owneroverlapcopy.hh>

/**
 * @file
 * @brief Functionality for redistributing a parallel index set using graph partitioning.
 *
 * @author Steffen Müthing
 */

namespace Dune {
  namespace ISTL {

#if HAVE_MPI

#ifndef DOXYGEN

    namespace impl {

      enum class CommGraphBasedMatrixRepartitionCommTags : int {

        mapped_row_indices = 11001,

      };

      template<
        typename Matrix,
        typename RemoteIndices
        >
      auto calculatePartitioning(
        MPI_Comm mpi_comm,
        int rank,
        int comm_size,
        int nparts,
        const Matrix& mat,
        const RemoteIndices& remote_indices,
        const ParameterTree& parameters
        )
      {

        auto local_node_data = std::array<idxtype,2>();
        auto& neighbour_count = local_node_data[0];
        neighbour_count = remote_indices.neighbours();
        auto& weight = local_node_data[1];

        auto weight_type = parameters.get("weightType","rows");

        if (weight_type == "rows")
          weight = mat.N();
        else if (weight_type == "nonzeros")
          weight = mat.nonzeroes();
        else
          DUNE_THROW(ISTLError,"Unknown weightType = '" << weight_type << "' in matrix repartitioning");

        // start sending the node data
        auto node_data = std::vector<decltype(local_node_data)>(rank == 0 ? comm_size : 0);
        MPI_Request node_data_request;
        MPI_Igather(
          &local_node_data,
          1,
          MPITraits<decltype(local_node_data)>::getType(),
          node_data.data(),
          1,
          MPITraits<decltype(local_node_data)>::getType(),
          0,
          mpi_comm,
          &node_data_request
          );

        auto local_edges = std::vector<std::array<idxtype,2> >();
        for(auto& neighbour : remote_indices)
          {
            auto neighbour_rank = neighbour.first;
            // use the sum of both index sets - IIUC, they should be identical, but this way,
            // we are on the safe side, and if they are identical, it won't change anything about
            // the result
            auto neighbour_weight = neighbour.second.first->size() + neighbour.second.second->size();
            local_edges.push_back({neighbour_rank,neighbour_weight});
          }

        std::sort(local_edges.begin(),local_edges.end());

        // wait for the node data to arrive
        MPI_Wait(&node_data_request, MPI_STATUS_IGNORE);

        auto node_partition_map = std::vector<idxtype>(comm_size);

        if (rank != 0)
          {
            // we just need to ship off the data to rank 0 and wait for the METIS results
            MPI_Gatherv(
              local_edges.data(),
              neighbour_count,
              MPITraits<std::decay_t<decltype(local_edges[0])>>::getType(),
              nullptr,
              nullptr,
              nullptr,
              MPI_DATATYPE_NULL,
              0,
              mpi_comm
              );
          }
        else
          {
            auto edge_counts = std::vector<idxtype>(comm_size);
            auto node_weights = std::vector<idxtype>(comm_size);
            for (int i = 0; i < comm_size; ++i)
              {
                edge_counts[i] = node_data[i][0];
                node_weights[i] = node_data[i][1];
              }
            auto edge_offsets = std::vector<idxtype>(comm_size + 1);
            edge_offsets[0] = 0;
            std::partial_sum(edge_counts.begin(),edge_counts.end(),edge_offsets.begin()+1);

            auto edge_target_nodes = std::vector<idxtype>(edge_offsets.back());
            auto edge_weights = std::vector<idxtype>(edge_offsets.back());

            {
              auto edges = decltype(local_edges)(edge_offsets.back());

              MPI_Gatherv(
                local_edges.data(),
                neighbour_count,
                MPITraits<std::decay_t<decltype(local_edges[0])>>::getType(),
                edges.data(),
                edge_counts.data(),
                edge_offsets.data(),
                MPITraits<std::decay_t<decltype(local_edges[0])>>::getType(),
                0,
                mpi_comm
                );

              for (std::size_t i = 0; i < edges.size(); ++i)
                {
                  edge_target_nodes[i] = edges[i][0];
                  edge_weights[i] = edges[i][1];
                }
            }

            idxtype node_count = comm_size;
            idxtype ncon = 1;
            idxtype partition_count = nparts;

            auto options = std::array<idxtype,METIS_NOPTIONS>{};
            METIS_SetDefaultOptions(options.data());
            options[METIS_OPTION_DBGLVL] = parameters.get("metis.dbglvl",0);

            idxtype result = 0;

            if (nparts > 1)
              {
                METIS_PartGraphKway(
                  &node_count,               // number of graph vertices
                  &ncon,                     // number of constraints
                  edge_offsets.data(),
                  edge_target_nodes.data(),
                  node_weights.data(),       // computational node weights
                  nullptr,                   // communication node weights
                  edge_weights.data(),
                  &partition_count,
                  nullptr,                   // we want an unweighted balancing
                  nullptr,                   // default tolerance for load imbalance
                  options.data(),
                  &result,
                  node_partition_map.data()
                  );
              }
            else
              {
                std::fill(node_partition_map.begin(),node_partition_map.end(),0);
              }

          }

        // broadcast the result
        MPI_Bcast(
          node_partition_map.data(),
          node_partition_map.size(),
          MPITraits<idxtype>::getType(),
          0,
          mpi_comm
          );

        if (parameters.get("metis.dbglvl",0) > 0)
          MPI_Barrier(mpi_comm);

        return std::move(node_partition_map);
      }

      template<
        typename OOComm,
        typename Matrix,
        typename RowMap,
        typename RowSizes,
        typename size_type,
        typename Sizes,
        typename PartitionMap,
        typename PartitionMembers
        >
      void condenseRows(
        OOComm& oocomm,
        int rank,
        MPI_Comm partition_comm,
        int partition_rank,
        int partition,
        const Matrix& mat,
        RowMap& condensed_row_to_old_row_map,
        RowMap& old_row_to_condensed_row_map,
        RowSizes& local_row_sizes,
        size_type& max_local_row_size,
        Sizes& condensed_sizes,
        Sizes& condensed_offsets,
        const PartitionMap& node_partition_map,
        const PartitionMembers& partition_members
        )
      {
        // condense out matrix entries that are only there due to communication with other partition members
        auto& index_set = oocomm.indexSet();
        auto& remote_indices = oocomm.remoteIndices();

        auto partition_size = partition_members.size();

        // markers for invalid entries in the maps
        constexpr auto INVALID_OWNER = std::numeric_limits<int>::max();
        constexpr auto INVALID_CONDENSED_OWNER = std::numeric_limits<int>::max() - 1;

        // map from rows to owning ranks (given in ranks of *old* partition)
        auto row_owners = std::vector<int>(mat.N(),INVALID_OWNER);

        // map from rows to owning ranks after condensing out intra-partition duplicates
        auto condensed_row_owners = std::vector<int>(mat.N(),INVALID_CONDENSED_OWNER);

        // flags to indicate whether a given row has been condensed out
        auto row_is_condensed = std::vector<bool>(mat.N(),false);

        // we construct the condensed set of rows by iterating over the remote index information and
        // applying the following rules:
        //
        // - if we own a row, we obviously keep it
        // - if a row is owned by another rank within our partition, we condense out the row and
        //   map it onto that rank
        // - if a row is owned by a rank that is not in our partition, we keep the proxy on the rank with
        //   the lowest index. On all other ranks, we condense out the row and map it to the row that we kept
        //   on the lowest rank.
        for (auto& neighbour : remote_indices)
          {
            auto remote_rank = neighbour.first;

            bool remote_rank_is_in_partition = node_partition_map[remote_rank] == partition;
            for (auto& index : *neighbour.second.second)
              {
                auto i = index.localIndexPair().local().local();
                bool owning_row = index.localIndexPair().local().attribute() == OwnerOverlapCopyAttributeSet::owner;
                if (row_owners[i] != rank and owning_row)
                  {
                    row_is_condensed[i] = false;
                    row_owners[i] = condensed_row_owners[i] = rank;
                  }
                switch (index.attribute())
                  {
                  case OwnerOverlapCopyAttributeSet::owner:
                    {
                      row_owners[i] = remote_rank;
                      // do not overwrite owner if we have already found a proxy within the partition
                      if (remote_rank_is_in_partition or not row_is_condensed[i])
                        {
                          row_is_condensed[i] = remote_rank_is_in_partition;
                          condensed_row_owners[i] = remote_rank;
                        }
                      break;
                    }
                  default:
                    {
                      if (remote_rank_is_in_partition)
                        {
                          if (row_is_condensed[i])
                            {
                              if (condensed_row_owners[i] != row_owners[i])
                                {
                                  using std::min;
                                  condensed_row_owners[i] = min(condensed_row_owners[i],remote_rank);
                                }
                            }
                          else
                            {
                              if (not owning_row and remote_rank < rank)
                                {
                                  condensed_row_owners[i] = remote_rank;
                                  row_is_condensed[i] = true;
                                }
                            }
                        }
                    }
                  }
              }
          }

        // replace placeholders with correct values for interior DOFs
        // by now, all rows that we do not own (and those that we own and share with other ranks)
        // have had their values overwritten with valid ranks, so all occurences of the special marker
        // indicate a row that is completely local to the current rank.
        std::replace(row_owners.begin(),row_owners.end(),INVALID_OWNER,rank);
        std::replace(condensed_row_owners.begin(),condensed_row_owners.end(),INVALID_CONDENSED_OWNER,rank);

        // now we need to build maps between the old set of rows and the new one
        // this requires communication within the partition to figure out the correct indices for rows that
        // were condensed out and mapped to other ranks within our partition
        // All of this communication is local to the current partition
        old_row_to_condensed_row_map.resize(mat.N());
        condensed_row_to_old_row_map.reserve(mat.N());

        using GlobalIndex = typename OOComm::GlobalID;

        // list of data that we expect to receive from each rank
        auto receive_lists = std::unordered_map<int,std::vector<std::pair<size_type,GlobalIndex>>>();

        constexpr auto INVALID_INDEX = std::numeric_limits<size_type>::max();

        oocomm.buildGlobalLookup(mat.N());
        auto& global_lookup = oocomm.globalLookup();

        for (size_type i = 0; i < row_owners.size(); ++i)
          if (row_is_condensed[i])
            {
              // if the row is condensed, look up the partition-local rank of the row owner and
              // add the expected information (row index in distributed matrix and associated global index)
              // to the asssociated receive list
              old_row_to_condensed_row_map[i] = INVALID_INDEX;
              auto partition_local_owner = std::lower_bound(
                partition_members.begin(),
                partition_members.end(),
                condensed_row_owners[i]
                ) - partition_members.begin();
              receive_lists[partition_local_owner].push_back(std::make_pair(i,global_lookup.pair(i)->global()));
            }
          else
            {
              // build index maps
              old_row_to_condensed_row_map[i] = condensed_row_to_old_row_map.size();
              condensed_row_to_old_row_map.push_back(i);
            }

        // in debug mode, we send the global id along with the (mapped) local index to be able to match up the
        // expected and received global indices. But sending all that additional data around is expensive, so
        // we skip it in normal operation. This requires a little work because we actually change around the data
        // structures, but it turns out that lambdas are perfect in this situation.
#if DUNE_ISTL_WITH_CHECKING
        using send_list_item_type = std::pair<GlobalIndex,size_type>;
        auto make_send_list_item = [&](auto& global_index, auto local_index)
          {
            return std::make_pair(index.localIndexPair().global(),old_row_to_condensed_row_map[local_index]);
          };
        auto get_received_local_index = [](auto& item)
          {
            return item->second;
          };
#else
        using send_list_item_type = size_type;
        auto make_send_list_item = [&](auto& global_index, auto local_index)
          {
            return old_row_to_condensed_row_map[local_index];
          };
        auto get_received_local_index = [](auto& item)
          {
            return item;
          };
#endif

        // per-rank list of data to send
        auto send_lists = std::unordered_map<int,std::vector<send_list_item_type>>();

        for (auto& neighbour : remote_indices)
          {
            auto remote_rank = neighbour.first;
            auto it = std::lower_bound(partition_members.begin(),partition_members.end(),remote_rank);

            if (it != partition_members.end() and *it == remote_rank) // remote rank is in partition
              {
                auto remote_rank_in_partition = it - partition_members.begin();
                for (auto& index : *neighbour.second.first)
                  {
                    auto i = index.localIndexPair().local().local();
                    // we need to send data if we either own the row or have been chosen as the partition-local
                    // proxy for a remote row
                    // in both cases, we send the condensed local index of the row
                    if (condensed_row_owners[i] == rank or node_partition_map[condensed_row_owners[i]] != partition)
                      {
                        send_lists[remote_rank_in_partition].push_back(make_send_list_item(index,i));
                      }
                  }
              }
          }

        // send all data
        for (auto& send_list : send_lists)
          {
            auto target = send_list.first;
            auto& list = send_list.second;
            MPI_Request req;
            MPI_Isend(
              list.data(),
              list.size(),
              MPITraits<send_list_item_type>::getType(),
              target,
              to_underlying(CommGraphBasedMatrixRepartitionCommTags::mapped_row_indices),
              partition_comm,
              &req
              );
            MPI_Request_free(&req);
          }

        // in order to communicate the matrix contents later on, we need to know the number
        // of matrix entries per row
        local_row_sizes.reserve(mat.N());
        // we also need the maximum number of entries per row for a later step
        max_local_row_size = 0;

        for (auto i : condensed_row_to_old_row_map)
          {
            size_type row_size = mat[i].size();
            local_row_sizes.push_back(row_size);
            using std::max;
            max_local_row_size = max(max_local_row_size,row_size);
          }

        // now we need to figure out the number of rows per rank after condensing out duplicates
        // so that we can (a) convert the rank-local indices to global ones and (b) gather data,
        // for which we need per-rank offsets and sizes
        condensed_sizes.resize(partition_size);
        auto condensed_size = local_row_sizes.size();

        MPI_Allgather(
          &condensed_size,
          1,
          MPITraits<int>::getType(),
          condensed_sizes.data(),
          1,
          MPITraits<int>::getType(),
          partition_comm
          );

        // convert sizes to offsets
        condensed_offsets.resize(partition_size + 1);
        condensed_offsets[0] = 0;
        std::partial_sum(condensed_sizes.begin(),condensed_sizes.end(),condensed_offsets.begin() + 1);

        // the global offset for our local rows
        auto global_offset = condensed_offsets[partition_rank];

        auto recv_buf = typename decltype(send_lists)::mapped_type();

        // sort the receive lists by global index, the send lists should already be sorted by it
        std::for_each(receive_lists.begin(),receive_lists.end(),[](auto& list)
                      {
                        std::sort(list.second.begin(),list.second.end(),[](auto& a, auto& b){
                            return a.second < b.second;
                          });
                      });

        // process remote index information. We do this by probing for new messages until
        // we have received data from all ranks that we expected
        for (size_type i = 0; i < receive_lists.size(); ++i)
          {
            MPI_Status status;
            MPI_Probe(
              MPI_ANY_SOURCE,
              to_underlying(CommGraphBasedMatrixRepartitionCommTags::mapped_row_indices),
              partition_comm,
              &status
              );

            // figure out who sent the data
            int source = status.MPI_SOURCE;

            // get the corresponding global offset
            size_type offset = condensed_offsets[source];

            // reserve memory and receive data
            int count;
            MPI_Get_count(&status,MPITraits<send_list_item_type>::getType(),&count);
            recv_buf.resize(count);

            MPI_Recv(
              recv_buf.data(),
              count,
              MPITraits<send_list_item_type>::getType(),
              source,
              to_underlying(CommGraphBasedMatrixRepartitionCommTags::mapped_row_indices),
              partition_comm,
              MPI_STATUS_IGNORE
              );

            auto& local_list = receive_lists[source];

            // make sure the local receive list has the same size as the data we got sent
            assert(local_list.size() == recv_buf.size());

            auto recv_it = recv_buf.begin();
            auto recv_end = recv_buf.end();
            auto local_it = local_list.begin();
            for (; recv_it != recv_end ; ++recv_it, ++local_it)
              {
#if DUNE_ISTL_WITH_CHECKING
                // make sure the entries actually match up
                assert(recv_it->first == local_it->second);
#endif
                // calculate correct row index for condensed out row within merged matrix
                old_row_to_condensed_row_map[local_it->first] = offset + get_received_local_index(*recv_it);
              }
          }

        // convert non-condensed row indices to globally unique mapping within partition
        for(auto i : condensed_row_to_old_row_map)
          old_row_to_condensed_row_map[i] += global_offset;
      }


      template<
        typename size_type,
        typename Matrix,
        typename RowMap,
        typename RowSizes,
        typename Sizes
        >
      std::array<MPI_Request,3> gatherMatrixData(
        MPI_Comm partition_comm,
        int partition_owner,
        bool owning_partition,
        size_type partition_size,
        const Matrix& mat,
        size_type max_local_row_size,
        Matrix& out_mat,
        const RowMap& condensed_row_to_old_row_map,
        const RowMap&  old_row_to_condensed_row_map,
        const RowSizes& local_row_sizes,
        const Sizes& condensed_sizes,
        const Sizes& condensed_offsets
        )
      {

        auto matrix_requests = std::array<MPI_Request,3>();

        // collect all row sizes on the owning rank to be able to allocate sufficient memory
        auto row_sizes = std::vector<size_type>(owning_partition ? condensed_offsets.back() : 0);

        MPI_Igatherv(
          local_row_sizes.data(),
          local_row_sizes.size(),
          MPITraits<size_type>::getType(),
          row_sizes.data(),condensed_sizes.data(),
          condensed_offsets.data(),
          MPITraits<size_type>::getType(),
          partition_owner,
          partition_comm,
          &matrix_requests[0]
          );

        // allocate data for matrix column indices and data
        auto local_condensed_nonzero_count = std::accumulate(local_row_sizes.begin(),local_row_sizes.end(),0);

        auto nonzero_sizes = std::vector<int>(owning_partition ? partition_size : 0);

        MPI_Gather(
          &local_condensed_nonzero_count, 1, MPITraits<int>::getType(),
          nonzero_sizes.data(), 1, MPITraits<int>::getType(),
          partition_owner,
          partition_comm
          );

        // the owner has to actually calculate the correct per-rank offsets
        auto nonzero_offsets = std::vector<int>(owning_partition ? partition_size + 1 : 0);
        if (owning_partition)
          {
            nonzero_offsets[0] = 0;
            std::partial_sum(nonzero_sizes.begin(),nonzero_sizes.end(),nonzero_offsets.begin() + 1);
          }
        auto global_condensed_nonzero_count = owning_partition ? nonzero_offsets.back() : 0;

        // extract condensed matrix data
        // this algorithm has been moved to a generic lambda as it will have to operate on different
        // types of iterators depending on whether we are on the owning rank or not
        auto permutation = std::vector<size_type>(max_local_row_size);
        auto extract_matrix_data = [&](auto col_index_it, auto data_it) {

          // Due to the row condensing, the column order in the merged matrix
          // might be different to our local order, so we have to permute the
          // columns

          // vector to hold permutation
          auto permutation_begin = permutation.begin();

          // iterate over all rows that we contribute to the merged matrix
          for (auto i : condensed_row_to_old_row_map)
            {
              auto old_row = mat[i];
              auto permutation_end = permutation_begin + old_row.size();
              std::iota(permutation_begin,permutation_end,0);

              // figure out correct permutation
              std::sort(permutation_begin,permutation_end,[&](auto a, auto b){
                  return old_row_to_condensed_row_map[old_row.index(a)] < old_row_to_condensed_row_map[old_row.index(b)];
                });

              // stream correctly permuted column indices and values into target buffer
              for (auto it = permutation_begin ; it != permutation_end ; ++it, ++col_index_it, ++data_it)
                {
                  *col_index_it = old_row_to_condensed_row_map[old_row.index(*it)];
                  *data_it = old_row.entry(*it);
                }
            }
        };

        // allocate send buffers - the partition owner directly writes into the target buffers within the new matrix
        auto col_indices = std::vector<size_type>(owning_partition ? 0 : local_condensed_nonzero_count);
        auto data = std::vector<typename Matrix::block_type>(owning_partition ? 0 : local_condensed_nonzero_count);

        if (not owning_partition)
          {
            // fill send buffers
            extract_matrix_data(col_indices.begin(),data.begin());

            // send column indices
            MPI_Igatherv(
              col_indices.data(),
              local_condensed_nonzero_count,
              MPITraits<size_type>::getType(),
              nullptr,
              nullptr,
              nullptr,
              MPI_DATATYPE_NULL,
              partition_owner,
              partition_comm,
              &matrix_requests[1]
              );

            // send values
            MPI_Igatherv(
              data.data(),
              local_condensed_nonzero_count,
              MPITraits<typename Matrix::block_type>::getType(),
              nullptr,
              nullptr,
              nullptr,
              MPI_DATATYPE_NULL,
              partition_owner,
              partition_comm,
              &matrix_requests[2]
              );
          }
        else
          {
            // set up the target matrix
            out_mat.setBuildMode(Matrix::random);
            out_mat.setSize(condensed_offsets.back(),condensed_offsets.back(),global_condensed_nonzero_count);

            // wait for the row sizes to arrive
            MPI_Wait(&matrix_requests[0],MPI_STATUS_IGNORE);

            // coerce the BCRS matrix into allocating the required memory
            for(size_type i = 0; i < row_sizes.size(); ++i)
              {
                out_mat.setrowsize(i,row_sizes[i]);
              }
            out_mat.endrowsizes();
            out_mat.endindices();

            // directly stream our local portion of the distributed matrix into the output matrix
            extract_matrix_data(columnIndices(out_mat) + nonzero_offsets[partition_owner], values(out_mat) + nonzero_offsets[partition_owner]);

            // receive column indices
            MPI_Igatherv(
              MPI_IN_PLACE,
              0,
              MPI_DATATYPE_NULL,
              columnIndices(out_mat),
              nonzero_sizes.data(),
              nonzero_offsets.data(),
              MPITraits<size_type>::getType(),
              partition_owner,
              partition_comm,
              &matrix_requests[1]
              );

            // receive values
            MPI_Igatherv(
              MPI_IN_PLACE,
              0,
              MPI_DATATYPE_NULL,
              values(out_mat),
              nonzero_sizes.data(),
              nonzero_offsets.data(),
              MPITraits<typename Matrix::block_type>::getType(),
              partition_owner,
              partition_comm,
              &matrix_requests[2]
              );
          }
        // we return these without waiting for them, as the communication can be overlapped
        // with the next step
        return matrix_requests;
      }


      template<
        typename size_type,
        typename RemoteIndices,
        typename RowMap,
        typename NodePartitionMap
        >
      std::vector<std::tuple<int,typename RemoteIndices::GlobalIndex,size_type,char,char>>
      gatherIndexData(
        MPI_Comm partition_comm,
        int partition,
        int partition_size,
        int partition_owner,
        bool owning_partition,
        const RemoteIndices& remote_indices,
        const RowMap& old_row_to_condensed_row_map,
        const NodePartitionMap& node_partition_map
        )
      {
        using GlobalIndex = typename RemoteIndices::GlobalIndex;

        // yeah, the following looks horrible - don't blame me, blame the RemoteIndices
        // in the tuple we store the following:
        // - remote partition rank
        // - global index
        // - condensed row index of the DOF
        // - local index attribute
        // - remote index attribute
        using remote_tuple_type = std::tuple<int,GlobalIndex,size_type,char,char>;

        // a vector with all remote indices connected to ranks outside the current partition
        auto cross_partition_indices = std::vector<remote_tuple_type>();

        for (auto& neighbour : remote_indices)
          {
            auto remote_rank = neighbour.first;
            bool remote_rank_is_in_partition = node_partition_map[remote_rank] == partition;
            if (remote_rank_is_in_partition)
              continue;

            for (auto& index : *neighbour.second.first)
              {
                auto i = index.localIndexPair().local().local();
                cross_partition_indices.emplace_back(
                  node_partition_map[remote_rank],
                  index.localIndexPair().global(),
                  old_row_to_condensed_row_map[i],
                  index.localIndexPair().local().attribute(),
                  index.attribute()
                  );
              }
          }

        // collect number of remote indices per rank
        auto remote_index_sizes = std::vector<int>(owning_partition ? partition_size : 0);
        int remote_index_size = cross_partition_indices.size();

        MPI_Gather(
          &remote_index_size,
          1,
          MPITraits<int>::getType(),
          remote_index_sizes.data(),
          1,
          MPITraits<int>::getType(),
          partition_owner,
          partition_comm
          );

        auto remote_index_list = std::vector<remote_tuple_type>();

        if (not owning_partition)
          {
            // we just interpret the remote_tuple_type as an array of char to avoid the expensive
            // MPI type logic for the rather complex structure of remote_tuple_type
            MPI_Gatherv(
              cross_partition_indices.data(),
              remote_index_size * sizeof(remote_tuple_type),
              MPITraits<char>::getType(),
              nullptr,
              nullptr,
              nullptr,
              MPI_DATATYPE_NULL,
              partition_owner,
              partition_comm
              );
          }
        else
          {
            auto remote_index_offsets = std::vector<int>(partition_size + 1);
            remote_index_offsets[0] = 0;

            // we need to scale the sizes by the size of the remote tuple type
            std::transform(
              remote_index_sizes.begin(),
              remote_index_sizes.end(),
              remote_index_sizes.begin(),
              [](auto size){
                return size * sizeof(remote_tuple_type);
              });

            // calculate offsets
            std::partial_sum(remote_index_sizes.begin(),remote_index_sizes.end(),remote_index_offsets.begin() + 1);

            // allocate receive buffer
            remote_index_list.resize(remote_index_offsets.back() / sizeof(remote_tuple_type));

            // collect data
            MPI_Gatherv(
              cross_partition_indices.data(),
              remote_index_size * sizeof(remote_tuple_type),
              MPITraits<char>::getType(),
              remote_index_list.data(),
              remote_index_sizes.data(),
              remote_index_offsets.data(),
              MPITraits<char>::getType(),
              partition_owner,
              partition_comm
              );
          }
        return std::move(remote_index_list);
      }

      template<
        typename size_type,
        typename OOComm,
        typename RemoteIndices
        >
      void buildIndexSetAndRemoteIndices(
        MPI_Comm result_comm,
        OOComm& oocomm,
        OOComm* out_comm,
        RemoteIndices& remote_indices
        )
      {
        using GlobalIndex = typename OOComm::GlobalID;
        using remote_tuple_type = typename RemoteIndices::value_type;

        // create OwnerOverlapCopyCommunication
        out_comm = new OOComm(result_comm,oocomm.getSolverCategory(),true);

        auto& comm_index_set = out_comm->indexSet();
        auto& comm_remote_indices = out_comm->remoteIndices();

        // sort the indices
        // Due to the order of the entries in the tuple, this will sort the entries first by the remote partition number
        // (which is the remote rank after repartitioning) and then by global index, which places all entries for a given
        // remote index in consecutive order
        std::sort(remote_indices.begin(),remote_indices.end());

        // some lambdas to make accessing the tuple entries more readable
        auto _partition = [](auto & t) { return std::get<0>(t); };
        // make sure to return a reference, as copying the global index might be expensive
        auto _global_index = [](auto & t) -> decltype(auto) { return std::get<1>(t); };
        auto _local_index = [](auto & t) { return std::get<2>(t); };
        auto _local_attribute = [](auto & t) { return std::get<3>(t); };
        auto _remote_attribute = [](auto & t) { return std::get<4>(t); };

        auto it = remote_indices.begin();
        auto global_end_it = remote_indices.end();

        auto global_indices = std::vector<std::tuple<size_type,GlobalIndex,char>>();

        // the remote indices want a set with all neighboring ranks
        auto neighbour_partitions = std::set<int>();

        // a map to store the data required to rebuild the per-rank remote index data
        auto neighbour_indices = std::map<
          int,
          std::vector<
            std::tuple<
              std::decay_t<decltype(_local_index(*it))>,
              std::decay_t<decltype(_local_attribute(*it))>,
              std::decay_t<decltype(_global_index(*it))>,
              std::decay_t<decltype(_remote_attribute(*it))>
              >
            >
          >();

        // this is a global loop over all remote index entries
        while (it != global_end_it)
          {
            // We want to iterate only over the entries associated with a single partition (i.e. remote rank)
            auto end_it = std::upper_bound(it,global_end_it,remote_tuple_type(_partition(*it),{0},0,0,0),[=](auto& a, auto& b){
                return _partition(a) < _partition(b);
              });

            // register as neighbour
            neighbour_partitions.insert(_partition(*it));

            while (it != end_it)
              {
                auto& global_index = _global_index(*it);

                // record index set entry
                global_indices.emplace_back(_local_index(*it),_global_index(*it),_local_attribute(*it));
                // record remote indices entry
                neighbour_indices[_partition(*it)].emplace_back(_local_index(*it),_local_attribute(*it),_global_index(*it),_remote_attribute(*it));

                // skip over all remaining entries for this global index - they will all have the same attribute
                for (; it != end_it and _global_index(*it) == global_index; ++it)
                  {}
              }
          }

        // create sorted list of global indices without duplicates
        std::sort(global_indices.begin(),global_indices.end(),[](auto& a, auto& b){
            return std::get<0>(a) < std::get<0>(b);
          });

        auto unique_global_indices = decltype(global_indices)();
        std::unique_copy(global_indices.begin(),global_indices.end(),std::back_inserter(unique_global_indices));

        // build index set
        comm_index_set.beginResize();

        using LocalIndex = typename std::decay_t<decltype(comm_index_set)>::LocalIndex;

        for (auto& i : unique_global_indices)
          {
            comm_index_set.add(
              std::get<1>(i),
              LocalIndex(
                std::get<0>(i),
                static_cast<Dune::OwnerOverlapCopyAttributeSet::AttributeSet>(std::get<2>(i)),
                true
                )
              );
          }

        // finish index set construction
        comm_index_set.endResize();
        comm_remote_indices.setNeighbours(neighbour_partitions);

        // build remote index information
        for (auto& i : neighbour_indices)
          {
            auto partition_index = i.first;
            auto& indices = i.second;

            // put data into both send and receive lists - maybe that's not necessary, but I don't really understand that
            // code enough to be sure...
            auto send_modifier = comm_remote_indices.template getModifier<false,true>(partition_index);
            auto receive_modifier = comm_remote_indices.template getModifier<false,false>(partition_index);

            // again, some lambdas to improve readability
            auto _global_index = [](auto & t) { return std::get<2>(t); };
            auto _remote_attribute = [](auto & t) { return static_cast<Dune::OwnerOverlapCopyAttributeSet::AttributeSet>(std::get<3>(t)); };

            for (auto& ri : indices)
              {
                using RemoteIndex = typename OOComm::RX;
                auto& index_pair = comm_index_set[_global_index(ri)];

                send_modifier.insert(RemoteIndex(_remote_attribute(ri),&index_pair));
                receive_modifier.insert(RemoteIndex(_remote_attribute(ri),&index_pair));
              }
          }

        // manually force the remote indices into a usable state
        comm_remote_indices.setBuilt();

      }

    } // namespace impl

#endif // DOXYGEN

    template<class M, class GlobalIndex, class T2, typename RI>
    bool commGraphBasedMatrixRepartitioning(
      const M& mat,
      Dune::OwnerOverlapCopyCommunication<GlobalIndex,T2>& oocomm,
      idxtype nparts,
      Dune::OwnerOverlapCopyCommunication<GlobalIndex,T2>*& out_comm,
      M& out_mat,
      RI& redistribute_information,
      const ParameterTree& parameters
      )
    {

      if(oocomm.communicator().rank()==0 && parameters.get("verbose",0) > 0)
        std::cout<<"Repartitioning from "<<oocomm.communicator().size()
                 <<" to "<<nparts<<" parts"<<std::endl;
      Timer time;
      auto rank = oocomm.communicator().rank();
      auto comm_size = oocomm.communicator().size();
      MPI_Comm mpi_comm = oocomm.communicator();

      using size_type = std::size_t;

      auto& remote_indices = oocomm.remoteIndices();

      // step 1: determine new partitioning

      auto node_partition_map = impl::calculatePartitioning(
        mpi_comm,
        rank,
        comm_size,
        nparts,
        mat,
        remote_indices,
        parameters
        );

      // step 2: set up partition data

      // figure out the inverse map and the partition owners
      auto partition = node_partition_map[rank];
      auto partition_members = std::vector<int>();

      for (int i = 0; i < comm_size; ++i)
        if (node_partition_map[i] == partition)
          partition_members.push_back(i);

      // member with highest rank becomes partition owner
      int partition_owner = partition_members.size() - 1;
      int partition_size = partition_members.size();

      // create per-partition communicators
      MPI_Comm partition_comm;
      MPI_Comm_split(mpi_comm, partition, rank, &partition_comm);

      int partition_rank = -1;
      MPI_Comm_rank(partition_comm,&partition_rank);
      bool owning_partition = partition_rank == partition_owner;

      // step 3: calculate compressed index set

      auto condensed_row_to_old_row_map = typename RI::SizeVector();
      auto old_row_to_condensed_row_map = typename RI::SizeVector();
      auto local_row_sizes = std::vector<size_type>();
      auto max_local_row_size = size_type(0);
      auto condensed_sizes = typename RI::IntVector();
      auto condensed_offsets = typename RI::IntVector();

      impl::condenseRows(
        oocomm,
        rank,
        partition_comm,
        partition_rank,
        partition,
        mat,
        condensed_row_to_old_row_map,
        old_row_to_condensed_row_map,
        local_row_sizes,
        max_local_row_size,
        condensed_sizes,
        condensed_offsets,
        node_partition_map,
        partition_members
        );

      // step 4: communicate matrix data

      auto matrix_requests = impl::gatherMatrixData<size_type>(
        partition_comm,
        partition_owner,
        owning_partition,
        partition_size,
        mat,
        max_local_row_size,
        out_mat,
        condensed_row_to_old_row_map,
        old_row_to_condensed_row_map,
        local_row_sizes,
        condensed_sizes,
        condensed_offsets
        );

      // step 5: communicate remote index data

      auto remote_index_data = impl::gatherIndexData<size_type>(
        partition_comm,
        partition,
        partition_size,
        partition_owner,
        owning_partition,
        remote_indices,
        old_row_to_condensed_row_map,
        node_partition_map
        );

      // step 6: create new communication object

      // the communicator for the repartitioned matrix
      MPI_Comm result_comm;
      if (not owning_partition)
        {
          // don't participate in the communicator for the repartitioned matrix
          MPI_Comm_split(mpi_comm,MPI_UNDEFINED,0,&result_comm);
        }
      else
        {
          // create communicator for repartitioned matrix
          MPI_Comm_split(mpi_comm,0,partition,&result_comm);

          // build remote index information and global index set from data
          // in remote_index_data
          impl::buildIndexSetAndRemoteIndices<size_type>(
            result_comm,
            oocomm,
            out_comm,
            remote_index_data
            );

          // wrap up the matrix transfer
          MPI_Waitall(3,matrix_requests.data(),MPI_STATUSES_IGNORE);

        }

      redistribute_information = RI(
        partition_comm,
        partition_owner,
        owning_partition,
        std::move(condensed_row_to_old_row_map),
        std::move(old_row_to_condensed_row_map),
        std::move(condensed_sizes),
        std::move(condensed_offsets)
        );

      return owning_partition;
    }


#endif // HAVE_MPI

  } // namespace ISTL
} // namespace Dune

#endif // DUNE_ISTL_COMMGRAPHBASEDMATRIXREPARTITIONING_HH
