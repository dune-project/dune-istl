// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GALERKIN_HH
#define DUNE_GALERKIN_HH

#include "aggregates.hh"
#include "pinfo.hh"
#include <dune/common/poolallocator.hh>
#include <dune/common/enumset.hh>
#include <set>
#include <limits>
#include <algorithm>

namespace Dune
{
  namespace Amg
  {
    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */
    /** @file
     * @author Markus Blatt
     * @brief Provides a class for building the galerkin product
     * based on a aggregation scheme.
     */

    template<class T>
    struct OverlapVertex
    {
      /**
       * @brief The aggregate descriptor.
       */
      typedef T Aggregate;

      /**
       * @brief The vertex descriptor.
       */
      typedef T Vertex;

      /**
       * @brief The aggregate the vertex belongs to.
       */
      Aggregate* aggregate;

      /**
       * @brief The vertex descriptor.
       */
      Vertex vertex;
    };



    /**
     * @brief Functor for building the sparsity pattern of the matrix
     * using examineConnectivity.
     */
    template<class M>
    class SparsityBuilder
    {
    public:
      /**
       * @brief Constructor.
       * @param matrix The matrix whose sparsity pattern we
       * should set up.
       */
      SparsityBuilder(M& matrix);

      void insert(const typename M::size_type& index);

      void operator++();

      std::size_t minRowSize();

      std::size_t maxRowSize();

      std::size_t sumRowSize();
      std::size_t index()
      {
        return row_.index();
      }
    private:
      /** @brief Create iterator for the current row. */
      typename M::CreateIterator row_;
      /** @brief The minim row size. */
      std::size_t minRowSize_;
      /** @brief The maximum row size. */
      std::size_t maxRowSize_;
      std::size_t sumRowSize_;
#ifdef DUNE_ISTL_WITH_CHECKING
      bool diagonalInserted;
#endif
    };

    class BaseGalerkinProduct
    {
    public:
      /**
       * @brief Calculate the galerkin product.
       * @param fine The fine matrix.
       * @param aggregates The aggregate mapping.
       * @param coarse The coarse Matrix.
       * @param pinfo Parallel information about the fine level.
       * @param copy The attribute set identifying the copy nodes of the graph.
       */
      template<class M, class V, class I, class O>
      void calculate(const M& fine, const AggregatesMap<V>& aggregates, M& coarse,
                     const I& pinfo, const O& copy);

    };

    template<class T>
    class GalerkinProduct
      : public BaseGalerkinProduct
    {
    public:
      typedef T ParallelInformation;

      /**
       * @brief Calculates the coarse matrix via a Galerkin product.
       * @param fineGraph The graph of the fine matrix.
       * @param visitedMap Map for marking vertices as visited.
       * @param pinfo Parallel information about the fine level.
       * @param aggregates The mapping of the fine level unknowns  onto aggregates.
       * @param size The number of columns and rows of the coarse matrix.
       * @param copy The attribute set identifying the copy nodes of the graph.
       */
      template<class G, class V, class Set>
      typename G::MutableMatrix* build(G& fineGraph, V& visitedMap,
                                       const ParallelInformation& pinfo,
                                       AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                       const typename G::Matrix::size_type& size,
                                       const Set& copy);
    private:

      /**
       * @brief Builds the data structure needed for rebuilding the aggregates int the overlap.
       * @param graph The graph of the matrix.
       * @param pinfo The parallel information.
       * @param aggregates The mapping onto the aggregates.
       */
      template<class G, class I, class Set>
      const OverlapVertex<typename G::VertexDescriptor>*
      buildOverlapVertices(const G& graph,  const I& pinfo,
                           AggregatesMap<typename G::VertexDescriptor>& aggregates,
                           const Set& overlap,
                           std::size_t& overlapCount);

      template<class A>
      struct OVLess
      {
        bool operator()(const OverlapVertex<A>& o1, const OverlapVertex<A>& o2)
        {
          return *o1.aggregate < *o2.aggregate;
        }
      };
    };

    template<>
    class GalerkinProduct<SequentialInformation>
      : public BaseGalerkinProduct
    {
    public:
      /**
       * @brief Calculates the coarse matrix via a Galerkin product.
       * @param fineGraph The graph of the fine matrix.
       * @param visitedMap Map for marking vertices as visited.
       * @param pinfo Parallel information about the fine level.
       * @param aggregates The mapping of the fine level unknowns  onto aggregates.
       * @param size The number of columns and rows of the coarse matrix.
       * @param copy The attribute set identifying the copy nodes of the graph.
       */
      template<class G, class V, class Set>
      typename G::MutableMatrix* build(G& fineGraph, V& visitedMap,
                                       const SequentialInformation& pinfo,
                                       const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                       const typename G::Matrix::size_type& size,
                                       const Set& copy);
    };

    struct BaseConnectivityConstructor
    {
      template<class R, class G, class V>
      static void constructOverlapConnectivity(R& row, G& graph, V& visitedMap,
                                               const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                               const OverlapVertex<typename G::VertexDescriptor>*& seed,
                                               const OverlapVertex<typename G::VertexDescriptor>* overlapEnd);

      /**
       * @brief Construct the connectivity of an aggregate in the overlap.
       */
      template<class R, class G, class V>
      static void constructNonOverlapConnectivity(R& row, G& graph, V& visitedMap,
                                                  const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                                  const typename G::VertexDescriptor& seed);


      /**
       * @brief Visitor for identifying connected aggregates during a breadthFirstSearch.
       */
      template<class G, class S, class V>
      class ConnectedBuilder
      {
      public:
        /**
         * @brief The type of the graph.
         */
        typedef G Graph;
        /**
         * @brief The constant edge iterator.
         */
        typedef typename Graph::ConstEdgeIterator ConstEdgeIterator;

        /**
         * @brief The type of the connected set.
         */
        typedef S Set;

        /**
         * @brief The type of the map for marking vertices as visited.
         */
        typedef V VisitedMap;

        /**
         * @brief The vertex descriptor of the graph.
         */
        typedef typename Graph::VertexDescriptor Vertex;

        /**
         * @brief Constructor
         * @param aggregates The mapping of the vertices onto the aggregates.
         * @param graph The graph to work on.
         * @param visitedMap The map for marking vertices as visited
         * @param connected The set to added the connected aggregates to.
         */
        ConnectedBuilder(const AggregatesMap<Vertex>& aggregates, Graph& graph,
                         VisitedMap& visitedMap, Set& connected);

        /**
         * @brief Process an edge pointing to another aggregate.
         * @param edge The iterator positioned at the edge.
         */
        void operator()(const ConstEdgeIterator& edge);

      private:
        /**
         * @brief The mapping of the vertices onto the aggregates.
         */
        const AggregatesMap<Vertex>& aggregates_;

        Graph& graph_;

        /**
         * @brief The map for marking vertices as visited.
         */
        VisitedMap& visitedMap_;

        /**
         * @brief The set to add the connected aggregates to.
         */
        Set& connected_;
      };

    };

    template<class G, class T>
    struct ConnectivityConstructor : public BaseConnectivityConstructor
    {
      typedef typename G::VertexDescriptor Vertex;

      template<class V, class O, class R>
      static void examine(G& graph,
                          V& visitedMap,
                          const T& pinfo,
                          const AggregatesMap<Vertex>& aggregates,
                          const O& overlap,
                          const OverlapVertex<Vertex>* overlapVertices,
                          const OverlapVertex<Vertex>* overlapEnd,
                          R& row);
    };

    template<class G>
    struct ConnectivityConstructor<G,SequentialInformation> : public BaseConnectivityConstructor
    {
      typedef typename G::VertexDescriptor Vertex;

      template<class V, class R>
      static void examine(G& graph,
                          V& visitedMap,
                          const SequentialInformation& pinfo,
                          const AggregatesMap<Vertex>& aggregates,
                          R& row);
    };

    template<class T>
    struct DirichletBoundarySetter
    {
      template<class M, class O>
      static void set(M& coarse, const T& pinfo, const O& copy);
    };

    template<>
    struct DirichletBoundarySetter<SequentialInformation>
    {
      template<class M, class O>
      static void set(M& coarse, const SequentialInformation& pinfo, const O& copy);
    };

    template<class R, class G, class V>
    void BaseConnectivityConstructor::constructNonOverlapConnectivity(R& row, G& graph, V& visitedMap,
                                                                      const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                                                      const typename G::VertexDescriptor& seed)
    {
      assert(row.index()==aggregates[seed]);
      row.insert(aggregates[seed]);
      ConnectedBuilder<G,R,V> conBuilder(aggregates, graph, visitedMap, row);
      typedef typename G::VertexDescriptor Vertex;
      typedef std::allocator<Vertex> Allocator;
      typedef SLList<Vertex,Allocator> VertexList;
      typedef typename AggregatesMap<Vertex>::DummyEdgeVisitor DummyVisitor;
      VertexList vlist;
      DummyVisitor dummy;
      aggregates.template breadthFirstSearch<true,false>(seed,aggregates[seed], graph, vlist, dummy,
                                                         conBuilder, visitedMap);
    }

    template<class R, class G, class V>
    void BaseConnectivityConstructor::constructOverlapConnectivity(R& row, G& graph, V& visitedMap,
                                                                   const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                                                   const OverlapVertex<typename G::VertexDescriptor>*& seed,
                                                                   const OverlapVertex<typename G::VertexDescriptor>* overlapEnd)
    {
      ConnectedBuilder<G,R,V> conBuilder(aggregates, graph, visitedMap, row);
      const typename G::VertexDescriptor aggregate=*seed->aggregate;

      if (row.index()==*seed->aggregate) {
        while(seed != overlapEnd && aggregate == *seed->aggregate) {
          row.insert(*seed->aggregate);
          // Walk over all neighbours and add them to the connected array.
          visitNeighbours(graph, seed->vertex, conBuilder);
          // Mark vertex as visited
          put(visitedMap, seed->vertex, true);
          ++seed;
        }
      }
    }

    template<class G, class S, class V>
    BaseConnectivityConstructor::ConnectedBuilder<G,S,V>::ConnectedBuilder(const AggregatesMap<Vertex>& aggregates,
                                                                           Graph& graph, VisitedMap& visitedMap,
                                                                           Set& connected)
      : aggregates_(aggregates), graph_(graph), visitedMap_(visitedMap), connected_(connected)
    {}

    template<class G, class S, class V>
    void BaseConnectivityConstructor::ConnectedBuilder<G,S,V>::operator()(const ConstEdgeIterator& edge)
    {
      const Vertex& vertex = aggregates_[edge.target()];
      assert(vertex!= AggregatesMap<Vertex>::UNAGGREGATED);
      if(vertex!= AggregatesMap<Vertex>::ISOLATED)
        connected_.insert(vertex);
    }

    template<class T>
    template<class G, class I, class Set>
    const OverlapVertex<typename G::VertexDescriptor>*
    GalerkinProduct<T>::buildOverlapVertices(const G& graph, const I& pinfo,
                                             AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                             const Set& overlap,
                                             std::size_t& overlapCount)
    {
      // count the overlap vertices.
      typedef typename G::ConstVertexIterator ConstIterator;
      typedef typename I::GlobalLookupIndexSet GlobalLookup;
      typedef typename GlobalLookup::IndexPair IndexPair;

      const ConstIterator end = graph.end();
      overlapCount = 0;

      const GlobalLookup& lookup=pinfo.globalLookup();

      for(ConstIterator vertex=graph.begin(); vertex != end; ++vertex) {
        const IndexPair* pair = lookup.pair(*vertex);

        if(pair!=0 && overlap.contains(pair->local().attribute()))
          ++overlapCount;
      }
      // Allocate space
      typedef typename G::VertexDescriptor Vertex;

      OverlapVertex<Vertex>* overlapVertices = new OverlapVertex<Vertex>[overlapCount=0 ? 1 : overlapCount];
      if(overlapCount==0)
        return overlapVertices;

      // Initialize them
      overlapCount=0;
      for(ConstIterator vertex=graph.begin(); vertex != end; ++vertex) {
        const IndexPair* pair = lookup.pair(*vertex);

        if(pair!=0 && overlap.contains(pair->local().attribute())) {
          overlapVertices[overlapCount].aggregate = &aggregates[pair->local()];
          overlapVertices[overlapCount].vertex = pair->local();
          ++overlapCount;
        }
      }

      dverb << overlapCount<<" overlap vertices"<<std::endl;

      std::sort(overlapVertices, overlapVertices+overlapCount, OVLess<Vertex>());
      // due to the sorting the isolated aggregates (to be skipped) are at the end.

      return overlapVertices;
    }

    template<class G, class T>
    template<class V, class O, class R>
    void ConnectivityConstructor<G,T>::examine(G& graph,
                                               V& visitedMap,
                                               const T& pinfo,
                                               const AggregatesMap<Vertex>& aggregates,
                                               const O& overlap,
                                               const OverlapVertex<Vertex>* overlapVertices,
                                               const OverlapVertex<Vertex>* overlapEnd,
                                               R& row)
    {
      typedef typename T::GlobalLookupIndexSet GlobalLookup;
      const GlobalLookup& lookup = pinfo.globalLookup();

      typedef typename G::VertexIterator VertexIterator;

      VertexIterator vend=graph.end();

#ifdef DUNE_ISTL_WITH_CHECKING
      std::set<Vertex> examined;
#endif

      // The aggregates owned by the process have lower local indices
      // then those not owned. We process them in the first pass.
      // They represent the rows 0, 1, ..., n of the coarse matrix
      for(VertexIterator vertex = graph.begin(); vertex != vend; ++vertex)
        if(!get(visitedMap, *vertex)) {
          // In the first pass we only process owner nodes
          typedef typename GlobalLookup::IndexPair IndexPair;
          const IndexPair* pair = lookup.pair(*vertex);
          if(pair==0 || !overlap.contains(pair->local().attribute())) {
#ifdef DUNE_ISTL_WITH_CHECKING
            assert(examined.find(aggregates[*vertex])==examined.end());
            examined.insert(aggregates[*vertex]);
#endif
            constructNonOverlapConnectivity(row, graph, visitedMap, aggregates, *vertex);

            // only needed for ALU
            // (ghosts with same global id as owners on the same process)
            if (SolverCategory::category(pinfo) == static_cast<int>(SolverCategory::nonoverlapping)) {
              if(overlapVertices != overlapEnd) {
                if(*overlapVertices->aggregate!=AggregatesMap<Vertex>::ISOLATED) {
                  constructOverlapConnectivity(row, graph, visitedMap, aggregates, overlapVertices, overlapEnd);
                }
                else{
                  ++overlapVertices;
                }
              }
            }
            ++row;
          }
        }

      dvverb<<"constructed "<<row.index()<<" non-overlapping rows"<<std::endl;

      // Now come the aggregates not owned by use.
      // They represent the rows n+1, ..., N
      while(overlapVertices != overlapEnd)
        if(*overlapVertices->aggregate!=AggregatesMap<Vertex>::ISOLATED) {

#ifdef DUNE_ISTL_WITH_CHECKING
          typedef typename GlobalLookup::IndexPair IndexPair;
          const IndexPair* pair = lookup.pair(overlapVertices->vertex);
          assert(pair!=0 && overlap.contains(pair->local().attribute()));
          assert(examined.find(aggregates[overlapVertices->vertex])==examined.end());
          examined.insert(aggregates[overlapVertices->vertex]);
#endif
          constructOverlapConnectivity(row, graph, visitedMap, aggregates, overlapVertices, overlapEnd);
          ++row;
        }else{
          ++overlapVertices;
        }
    }

    template<class G>
    template<class V, class R>
    void ConnectivityConstructor<G,SequentialInformation>::examine(G& graph,
                                                                   V& visitedMap,
                                                                   [[maybe_unused]] const SequentialInformation& pinfo,
                                                                   const AggregatesMap<Vertex>& aggregates,
                                                                   R& row)
    {
      typedef typename G::VertexIterator VertexIterator;

      VertexIterator vend=graph.end();
      for(VertexIterator vertex = graph.begin(); vertex != vend; ++vertex) {
        if(!get(visitedMap, *vertex)) {
          constructNonOverlapConnectivity(row, graph, visitedMap, aggregates, *vertex);
          ++row;
        }
      }

    }

    template<class M>
    SparsityBuilder<M>::SparsityBuilder(M& matrix)
      : row_(matrix.createbegin()),
        minRowSize_(std::numeric_limits<std::size_t>::max()),
        maxRowSize_(0), sumRowSize_(0)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      diagonalInserted = false;
#endif
    }
    template<class M>
    std::size_t SparsityBuilder<M>::maxRowSize()
    {
      return maxRowSize_;
    }
    template<class M>
    std::size_t SparsityBuilder<M>::minRowSize()
    {
      return minRowSize_;
    }

    template<class M>
    std::size_t SparsityBuilder<M>::sumRowSize()
    {
      return sumRowSize_;
    }
    template<class M>
    void SparsityBuilder<M>::operator++()
    {
      sumRowSize_ += row_.size();
      minRowSize_=std::min(minRowSize_, row_.size());
      maxRowSize_=std::max(maxRowSize_, row_.size());
      ++row_;
#ifdef DUNE_ISTL_WITH_CHECKING
      assert(diagonalInserted);
      diagonalInserted = false;
#endif
    }

    template<class M>
    void SparsityBuilder<M>::insert(const typename M::size_type& index)
    {
      row_.insert(index);
#ifdef DUNE_ISTL_WITH_CHECKING
      diagonalInserted = diagonalInserted || row_.index()==index;
#endif
    }

    template<class T>
    template<class G, class V, class Set>
    typename G::MutableMatrix*
    GalerkinProduct<T>::build(G& fineGraph, V& visitedMap,
                              const ParallelInformation& pinfo,
                              AggregatesMap<typename G::VertexDescriptor>& aggregates,
                              const typename G::Matrix::size_type& size,
                              const Set& overlap)
    {
      typedef OverlapVertex<typename G::VertexDescriptor> OverlapVertex;

      std::size_t count;

      const OverlapVertex* overlapVertices = buildOverlapVertices(fineGraph,
                                                                  pinfo,
                                                                  aggregates,
                                                                  overlap,
                                                                  count);
      typedef typename G::MutableMatrix M;
      M* coarseMatrix = new M(size, size, M::row_wise);

      // Reset the visited flags of all vertices.
      // As the isolated nodes will be skipped we simply mark them as visited

      typedef typename G::VertexIterator Vertex;
      Vertex vend = fineGraph.end();
      for(Vertex vertex = fineGraph.begin(); vertex != vend; ++vertex) {
        assert(aggregates[*vertex] != AggregatesMap<typename G::VertexDescriptor>::UNAGGREGATED);
        put(visitedMap, *vertex, aggregates[*vertex]==AggregatesMap<typename G::VertexDescriptor>::ISOLATED);
      }

      typedef typename G::MutableMatrix M;
      SparsityBuilder<M> sparsityBuilder(*coarseMatrix);

      ConnectivityConstructor<G,T>::examine(fineGraph, visitedMap, pinfo,
                                            aggregates, overlap,
                                            overlapVertices,
                                            overlapVertices+count,
                                            sparsityBuilder);

      dinfo<<pinfo.communicator().rank()<<": Matrix ("<<coarseMatrix->N()<<"x"<<coarseMatrix->M()<<" row: min="<<sparsityBuilder.minRowSize()<<" max="
           <<sparsityBuilder.maxRowSize()<<" avg="
           <<static_cast<double>(sparsityBuilder.sumRowSize())/coarseMatrix->N()
           <<std::endl;

      delete[] overlapVertices;

      return coarseMatrix;
    }

    template<class G, class V, class Set>
    typename G::MutableMatrix*
    GalerkinProduct<SequentialInformation>::build(G& fineGraph, V& visitedMap,
                                                  const SequentialInformation& pinfo,
                                                  const AggregatesMap<typename G::VertexDescriptor>& aggregates,
                                                  const typename G::Matrix::size_type& size,
                                                  [[maybe_unused]] const Set& overlap)
    {
      typedef typename G::MutableMatrix M;
      M* coarseMatrix = new M(size, size, M::row_wise);

      // Reset the visited flags of all vertices.
      // As the isolated nodes will be skipped we simply mark them as visited

      typedef typename G::VertexIterator Vertex;
      Vertex vend = fineGraph.end();
      for(Vertex vertex = fineGraph.begin(); vertex != vend; ++vertex) {
        assert(aggregates[*vertex] != AggregatesMap<typename G::VertexDescriptor>::UNAGGREGATED);
        put(visitedMap, *vertex, aggregates[*vertex]==AggregatesMap<typename G::VertexDescriptor>::ISOLATED);
      }

      SparsityBuilder<M> sparsityBuilder(*coarseMatrix);

      ConnectivityConstructor<G,SequentialInformation>::examine(fineGraph, visitedMap, pinfo,
                                                                aggregates, sparsityBuilder);
      dinfo<<"Matrix row: min="<<sparsityBuilder.minRowSize()<<" max="
           <<sparsityBuilder.maxRowSize()<<" average="
           <<static_cast<double>(sparsityBuilder.sumRowSize())/coarseMatrix->N()<<std::endl;
      return coarseMatrix;
    }

    template<class M, class V, class P, class O>
    void BaseGalerkinProduct::calculate(const M& fine, const AggregatesMap<V>& aggregates, M& coarse,
                                        const P& pinfo, [[maybe_unused]] const O& copy)
    {
      coarse = static_cast<typename M::field_type>(0);

      typedef typename M::ConstIterator RowIterator;
      RowIterator endRow = fine.end();

      for(RowIterator row = fine.begin(); row != endRow; ++row)
        if(aggregates[row.index()] != AggregatesMap<V>::ISOLATED) {
          assert(aggregates[row.index()]!=AggregatesMap<V>::UNAGGREGATED);
          typedef typename M::ConstColIterator ColIterator;
          ColIterator endCol = row->end();

          for(ColIterator col = row->begin(); col != endCol; ++col)
            if(aggregates[col.index()] != AggregatesMap<V>::ISOLATED) {
              assert(aggregates[row.index()]!=AggregatesMap<V>::UNAGGREGATED);
              coarse[aggregates[row.index()]][aggregates[col.index()]]+=*col;
            }
        }

      // get the right diagonal matrix values on copy lines from owner processes
      typedef typename M::block_type BlockType;
      std::vector<BlockType> rowsize(coarse.N(),BlockType(0));
      for (RowIterator row = coarse.begin(); row != coarse.end(); ++row)
        rowsize[row.index()]=coarse[row.index()][row.index()];
      pinfo.copyOwnerToAll(rowsize,rowsize);
      for (RowIterator row = coarse.begin(); row != coarse.end(); ++row)
        coarse[row.index()][row.index()] = rowsize[row.index()];

      // don't set dirichlet boundaries for copy lines to make novlp case work,
      // the preconditioner yields slightly different results now.

      // Set the dirichlet border
      //DirichletBoundarySetter<P>::template set<M>(coarse, pinfo, copy);

    }

    template<class T>
    template<class M, class O>
    void DirichletBoundarySetter<T>::set(M& coarse, const T& pinfo, const O& copy)
    {
      typedef typename T::ParallelIndexSet::const_iterator ConstIterator;
      ConstIterator end = pinfo.indexSet().end();
      typedef typename M::block_type Block;
      Block identity=Block(0.0);
      for(typename Block::RowIterator b=identity.begin(); b !=  identity.end(); ++b)
        b->operator[](b.index())=1.0;

      for(ConstIterator index = pinfo.indexSet().begin();
          index != end; ++index) {
        if(copy.contains(index->local().attribute())) {
          typedef typename M::ColIterator ColIterator;
          typedef typename M::row_type Row;
          Row row = coarse[index->local()];
          ColIterator cend = row.find(index->local());
          ColIterator col  = row.begin();
          for(; col != cend; ++col)
            *col = 0;

          cend = row.end();

          assert(col != cend); // There should be a diagonal entry
          *col = identity;

          for(++col; col != cend; ++col)
            *col = 0;
        }
      }
    }

    template<class M, class O>
    void DirichletBoundarySetter<SequentialInformation>::set(M& coarse,
                                                             const SequentialInformation& pinfo,
                                                             const O& overlap)
    {}

  } // namespace Amg
} // namespace Dune
#endif
