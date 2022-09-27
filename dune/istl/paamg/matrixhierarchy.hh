// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_MATRIXHIERARCHY_HH
#define DUNE_AMG_MATRIXHIERARCHY_HH

#include <algorithm>
#include <tuple>
#include "aggregates.hh"
#include "graph.hh"
#include "galerkin.hh"
#include "renumberer.hh"
#include "graphcreator.hh"
#include "hierarchy.hh"
#include <dune/istl/bvector.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/matrixutils.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/paamg/dependency.hh>
#include <dune/istl/paamg/graph.hh>
#include <dune/istl/paamg/indicescoarsener.hh>
#include <dune/istl/paamg/globalaggregates.hh>
#include <dune/istl/paamg/construction.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/paamg/transfer.hh>

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
     * @brief Provides a classes representing the hierarchies in AMG.
     */
    enum {
      /**
       * @brief Hard limit for the number of processes allowed.
       *
       * This is needed to prevent overflows when calculating
       * the coarsening rate. Currently set 72,000 which is
       * enough for JUGENE.
       */
      MAX_PROCESSES = 72000
    };

    /**
     * @brief The hierarchies build by the coarsening process.
     *
     * Namely a hierarchy of matrices, index sets, remote indices,
     * interfaces and communicators.
     */
    template<class M, class PI, class A=std::allocator<M> >
    class MatrixHierarchy
    {
    public:
      /** @brief The type of the matrix operator. */
      typedef M MatrixOperator;

      /** @brief The type of the matrix. */
      typedef typename MatrixOperator::matrix_type Matrix;

      /** @brief The type of the index set. */
      typedef PI ParallelInformation;

      /** @brief The allocator to use. */
      typedef A Allocator;

      /** @brief The type of the aggregates map we use. */
      typedef Dune::Amg::AggregatesMap<typename MatrixGraph<Matrix>::VertexDescriptor> AggregatesMap;

      /** @brief The type of the parallel matrix hierarchy. */
      typedef Dune::Amg::Hierarchy<MatrixOperator,Allocator> ParallelMatrixHierarchy;

      /** @brief The type of the parallel informarion hierarchy. */
      typedef Dune::Amg::Hierarchy<ParallelInformation,Allocator> ParallelInformationHierarchy;

      /** @brief Allocator for pointers. */
      using AAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<AggregatesMap*>;

      /** @brief The type of the aggregates maps list. */
      typedef std::list<AggregatesMap*,AAllocator> AggregatesMapList;

      /** @brief The type of the redistribute information. */
      typedef RedistributeInformation<ParallelInformation> RedistributeInfoType;

      /** @brief Allocator for RedistributeInfoType. */
      using RILAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<RedistributeInfoType>;

      /** @brief The type of the list of redistribute information. */
      typedef std::list<RedistributeInfoType,RILAllocator> RedistributeInfoList;

      /**
       * @brief Constructor
       * @param fineMatrix The matrix to coarsen.
       * @param pinfo The information about the parallel data decomposition at the first level.
       */
      MatrixHierarchy(std::shared_ptr<MatrixOperator> fineMatrix,
        std::shared_ptr<ParallelInformation> pinfo = std::make_shared<ParallelInformation>());

      ~MatrixHierarchy();

      /**
       * @brief Build the matrix hierarchy using aggregation.
       *
       * @brief criterion The criterion describing the aggregation process.
       */
      template<typename O, typename T>
      void build(const T& criterion);

      /**
       * @brief Recalculate the galerkin products.
       *
       * If the data of the fine matrix changes but not its sparsity pattern
       * this will recalculate all coarser levels without starting the expensive
       * aggregation process all over again.
       */
      template<class F>
      void recalculateGalerkin(const F& copyFlags);

      /**
       * @brief Coarsen the vector hierarchy according to the matrix hierarchy.
       * @param hierarchy The vector hierarchy to coarsen.
       */
      template<class V, class BA, class TA>
      void coarsenVector(Hierarchy<BlockVector<V,BA>, TA>& hierarchy) const;

      /**
       * @brief Coarsen the smoother hierarchy according to the matrix hierarchy.
       * @param smoothers The smoother hierarchy to coarsen.
       * @param args The arguments for the construction of the coarse level smoothers.
       */
      template<class S, class TA>
      void coarsenSmoother(Hierarchy<S,TA>& smoothers,
                           const typename SmootherTraits<S>::Arguments& args) const;

      /**
       * @brief Get the number of levels in the hierarchy.
       * @return The number of levels.
       */
      std::size_t levels() const;

      /**
       * @brief Get the max number of levels in the hierarchy of processors.
       * @return The maximum number of levels.
       */
      std::size_t maxlevels() const;

      bool hasCoarsest() const;

      /**
       * @brief Whether the hierarchy was built.
       * @return true if the MatrixHierarchy::build method was called.
       */
      bool isBuilt() const;

      /**
       * @brief Get the matrix hierarchy.
       * @return The matrix hierarchy.
       */
      const ParallelMatrixHierarchy& matrices() const;

      /**
       * @brief Get the hierarchy of the parallel data distribution information.
       * @return The hierarchy of the parallel data distribution information.
       */
      const ParallelInformationHierarchy& parallelInformation() const;

      /**
       * @brief Get the hierarchy of the mappings of the nodes onto aggregates.
       * @return The hierarchy of the mappings of the nodes onto aggregates.
       */
      const AggregatesMapList& aggregatesMaps() const;

      /**
       * @brief Get the hierarchy of the information about redistributions,
       * @return The hierarchy of the information about redistributions of the
       * data to fewer processes.
       */
      const RedistributeInfoList& redistributeInformation() const;

      double getProlongationDampingFactor() const
      {
        return prolongDamp_;
      }

      /**
       * @brief Get the mapping of fine level unknowns to coarse level
       * aggregates.
       *
       * For each fine level unknown i the correcponding data[i] is the
       * aggregate it belongs to on the coarsest level.
       *
       * @param[out] data The mapping of fine level unknowns to coarse level
       * aggregates.
       */
      void getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data) const;

    private:
      typedef typename ConstructionTraits<MatrixOperator>::Arguments MatrixArgs;
      typedef typename ConstructionTraits<ParallelInformation>::Arguments CommunicationArgs;
      /** @brief The list of aggregates maps. */
      AggregatesMapList aggregatesMaps_;
      /** @brief The list of redistributes. */
      RedistributeInfoList redistributes_;
      /** @brief The hierarchy of parallel matrices. */
      ParallelMatrixHierarchy matrices_;
      /** @brief The hierarchy of the parallel information. */
      ParallelInformationHierarchy parallelInformation_;

      /** @brief Whether the hierarchy was built. */
      bool built_;

      /** @brief The maximum number of level across all processors.*/
      int maxlevels_;

      double prolongDamp_;

      /**
       * @brief functor to print matrix statistics.
       */
      template<class Matrix, bool print>
      struct MatrixStats
      {

        /**
         * @brief Print matrix statistics.
         */
        static void stats([[maybe_unused]] const Matrix& matrix)
        {}
      };

      template<class Matrix>
      struct MatrixStats<Matrix,true>
      {
        struct calc
        {
          typedef typename Matrix::size_type size_type;
          typedef typename Matrix::row_type matrix_row;

          calc()
          {
            min=std::numeric_limits<size_type>::max();
            max=0;
            sum=0;
          }

          void operator()(const matrix_row& row)
          {
            min=std::min(min, row.size());
            max=std::max(max, row.size());
            sum += row.size();
          }

          size_type min;
          size_type max;
          size_type sum;
        };
        /**
         * @brief Print matrix statistics.
         */
        static void stats(const Matrix& matrix)
        {
          calc c= for_each(matrix.begin(), matrix.end(), calc());
          dinfo<<"Matrix row: min="<<c.min<<" max="<<c.max
               <<" average="<<static_cast<double>(c.sum)/matrix.N()
               <<std::endl;
        }
      };
    };

    /**
     * @brief The criterion describing the stop criteria for the coarsening process.
     */
    template<class T>
    class CoarsenCriterion : public T
    {
    public:
      /**
       * @brief The criterion for tagging connections as strong and nodes as isolated.
       * This might be e.g. SymmetricCriterion or UnSymmetricCriterion.
       */
      typedef T AggregationCriterion;

      /**
       * @brief Constructor
       * @param maxLevel The maximum number of levels allowed in the matrix hierarchy (default: 100).
       * @param coarsenTarget If the number of nodes in the matrix is below this threshold the
       * coarsening will stop (default: 1000).
       * @param minCoarsenRate If the coarsening rate falls below this threshold the
       * coarsening will stop (default: 1.2)
       * @param prolongDamp The damping factor to apply to the prolongated update (default: 1.6)
       * @param accumulate Whether to accumulate the data onto fewer processors on coarser levels.
       */
      CoarsenCriterion(int maxLevel=100, int coarsenTarget=1000, double minCoarsenRate=1.2,
                       double prolongDamp=1.6, AccumulationMode accumulate=successiveAccu)
        : AggregationCriterion(Dune::Amg::Parameters(maxLevel, coarsenTarget, minCoarsenRate, prolongDamp, accumulate))
      {}

      CoarsenCriterion(const Dune::Amg::Parameters& parms)
        : AggregationCriterion(parms)
      {}

    };

    template<typename M, typename C1>
    bool repartitionAndDistributeMatrix([[maybe_unused]] const M& origMatrix,
                                        [[maybe_unused]] std::shared_ptr<M> newMatrix,
                                        [[maybe_unused]] SequentialInformation& origComm,
                                        [[maybe_unused]] std::shared_ptr<SequentialInformation>& newComm,
                                        [[maybe_unused]] RedistributeInformation<SequentialInformation>& ri,
                                        [[maybe_unused]] int nparts,
                                        [[maybe_unused]] C1& criterion)
    {
      DUNE_THROW(NotImplemented, "Redistribution does not make sense in sequential code!");
    }


    template<typename M, typename C, typename C1>
    bool repartitionAndDistributeMatrix(const M& origMatrix,
                                        std::shared_ptr<M> newMatrix,
                                        C& origComm,
                                        std::shared_ptr<C>& newComm,
                                        RedistributeInformation<C>& ri,
                                        int nparts, C1& criterion)
    {
      Timer time;
#ifdef AMG_REPART_ON_COMM_GRAPH
      // Done not repartition the matrix graph, but a graph of the communication scheme.
      bool existentOnRedist=Dune::commGraphRepartition(origMatrix, origComm, nparts, newComm,
                                                       ri.getInterface(),
                                                       criterion.debugLevel()>1);

#else
      typedef Dune::Amg::MatrixGraph<const M> MatrixGraph;
      typedef Dune::Amg::PropertiesGraph<MatrixGraph,
          VertexProperties,
          EdgeProperties,
          IdentityMap,
          IdentityMap> PropertiesGraph;
      MatrixGraph graph(origMatrix);
      PropertiesGraph pgraph(graph);
      buildDependency(pgraph, origMatrix, criterion, false);

#ifdef DEBUG_REPART
      if(origComm.communicator().rank()==0)
        std::cout<<"Original matrix"<<std::endl;
      origComm.communicator().barrier();
      printGlobalSparseMatrix(origMatrix, origComm, std::cout);
#endif
      bool existentOnRedist=Dune::graphRepartition(pgraph, origComm, nparts,
                                                   newComm, ri.getInterface(),
                                                   criterion.debugLevel()>1);
#endif // if else AMG_REPART

      if(origComm.communicator().rank()==0  && criterion.debugLevel()>1)
        std::cout<<"Repartitioning took "<<time.elapsed()<<" seconds."<<std::endl;

      ri.setSetup();

#ifdef DEBUG_REPART
      ri.checkInterface(origComm.indexSet(), newComm->indexSet(), origComm.communicator());
#endif

      redistributeMatrix(const_cast<M&>(origMatrix), *newMatrix, origComm, *newComm, ri);

#ifdef DEBUG_REPART
      if(origComm.communicator().rank()==0)
        std::cout<<"Original matrix"<<std::endl;
      origComm.communicator().barrier();
      if(newComm->communicator().size()>0)
        printGlobalSparseMatrix(*newMatrix, *newComm, std::cout);
      origComm.communicator().barrier();
#endif

      if(origComm.communicator().rank()==0  && criterion.debugLevel()>1)
        std::cout<<"Redistributing matrix took "<<time.elapsed()<<" seconds."<<std::endl;
      return existentOnRedist;

    }

    template<class M, class IS, class A>
    MatrixHierarchy<M,IS,A>::MatrixHierarchy(std::shared_ptr<MatrixOperator> fineMatrix,
                                             std::shared_ptr<ParallelInformation> pinfo)
      : matrices_(fineMatrix),
        parallelInformation_(pinfo)
    {
      if (SolverCategory::category(*fineMatrix) != SolverCategory::category(*pinfo))
        DUNE_THROW(ISTLError, "MatrixOperator and ParallelInformation must belong to the same category!");
    }

    template<class M, class IS, class A>
    template<typename O, typename T>
    void MatrixHierarchy<M,IS,A>::build(const T& criterion)
    {
      prolongDamp_ = criterion.getProlongationDampingFactor();
      typedef O OverlapFlags;
      typedef typename ParallelMatrixHierarchy::Iterator MatIterator;
      typedef typename ParallelInformationHierarchy::Iterator PInfoIterator;

      static const int noints=(Dune::Amg::MAX_PROCESSES/4096>0) ? (Dune::Amg::MAX_PROCESSES/4096) : 1;

      typedef bigunsignedint<sizeof(int)*8*noints> BIGINT;
      GalerkinProduct<ParallelInformation> productBuilder;
      MatIterator mlevel = matrices_.finest();
      MatrixStats<typename M::matrix_type,MINIMAL_DEBUG_LEVEL<=INFO_DEBUG_LEVEL>::stats(mlevel->getmat());

      PInfoIterator infoLevel = parallelInformation_.finest();
      BIGINT finenonzeros=countNonZeros(mlevel->getmat());
      finenonzeros = infoLevel->communicator().sum(finenonzeros);
      BIGINT allnonzeros = finenonzeros;


      int level = 0;
      int rank = 0;

      BIGINT unknowns = mlevel->getmat().N();

      unknowns = infoLevel->communicator().sum(unknowns);
      double dunknowns=unknowns.todouble();
      infoLevel->buildGlobalLookup(mlevel->getmat().N());
      redistributes_.push_back(RedistributeInfoType());

      for(; level < criterion.maxLevel(); ++level, ++mlevel) {
        assert(matrices_.levels()==redistributes_.size());
        rank = infoLevel->communicator().rank();
        if(rank==0 && criterion.debugLevel()>1)
          std::cout<<"Level "<<level<<" has "<<dunknowns<<" unknowns, "<<dunknowns/infoLevel->communicator().size()
                   <<" unknowns per proc (procs="<<infoLevel->communicator().size()<<")"<<std::endl;

        MatrixOperator* matrix=&(*mlevel);
        ParallelInformation* info =&(*infoLevel);

        if((
#if HAVE_PARMETIS
             criterion.accumulate()==successiveAccu
#else
             false
#endif
             || (criterion.accumulate()==atOnceAccu
                 && dunknowns < 30*infoLevel->communicator().size()))
           && infoLevel->communicator().size()>1 &&
           dunknowns/infoLevel->communicator().size() <= criterion.coarsenTarget())
        {
          // accumulate to fewer processors
          std::shared_ptr<Matrix> redistMat = std::make_shared<Matrix>();
          std::shared_ptr<ParallelInformation> redistComm;
          std::size_t nodomains = (std::size_t)std::ceil(dunknowns/(criterion.minAggregateSize()
                                                                    *criterion.coarsenTarget()));
          if( nodomains<=criterion.minAggregateSize()/2 ||
              dunknowns <= criterion.coarsenTarget() )
            nodomains=1;

          bool existentOnNextLevel =
            repartitionAndDistributeMatrix(mlevel->getmat(), redistMat, *infoLevel,
                                           redistComm, redistributes_.back(), nodomains,
                                           criterion);
          BIGINT unknownsRedist = redistMat->N();
          unknownsRedist = infoLevel->communicator().sum(unknownsRedist);
          dunknowns= unknownsRedist.todouble();
          if(redistComm->communicator().rank()==0 && criterion.debugLevel()>1)
            std::cout<<"Level "<<level<<" (redistributed) has "<<dunknowns<<" unknowns, "<<dunknowns/redistComm->communicator().size()
                     <<" unknowns per proc (procs="<<redistComm->communicator().size()<<")"<<std::endl;
          MatrixArgs args(redistMat, *redistComm);
          mlevel.addRedistributed(ConstructionTraits<MatrixOperator>::construct(args));
          assert(mlevel.isRedistributed());
          infoLevel.addRedistributed(redistComm);
          infoLevel->freeGlobalLookup();

          if(!existentOnNextLevel)
            // We do not hold any data on the redistributed partitioning
            break;

          // Work on the redistributed Matrix from now on
          matrix = &(mlevel.getRedistributed());
          info = &(infoLevel.getRedistributed());
          info->buildGlobalLookup(matrix->getmat().N());
        }

        rank = info->communicator().rank();
        if(dunknowns <= criterion.coarsenTarget())
          // No further coarsening needed
          break;

        typedef PropertiesGraphCreator<MatrixOperator,ParallelInformation> GraphCreator;
        typedef typename GraphCreator::PropertiesGraph PropertiesGraph;
        typedef typename GraphCreator::GraphTuple GraphTuple;

        typedef typename PropertiesGraph::VertexDescriptor Vertex;

        std::vector<bool> excluded(matrix->getmat().N(), false);

        GraphTuple graphs = GraphCreator::create(*matrix, excluded, *info, OverlapFlags());

        AggregatesMap* aggregatesMap=new AggregatesMap(std::get<1>(graphs)->maxVertex()+1);

        aggregatesMaps_.push_back(aggregatesMap);

        Timer watch;
        watch.reset();
        auto [noAggregates, isoAggregates, oneAggregates, skippedAggregates] =
          aggregatesMap->buildAggregates(matrix->getmat(), *(std::get<1>(graphs)), criterion, level==0);

        if(rank==0 && criterion.debugLevel()>2)
          std::cout<<" Have built "<<noAggregates<<" aggregates totally ("<<isoAggregates<<" isolated aggregates, "<<
          oneAggregates<<" aggregates of one vertex,  and skipped "<<
          skippedAggregates<<" aggregates)."<<std::endl;
#ifdef TEST_AGGLO
        {
          // calculate size of local matrix in the distributed direction
          int start, end, overlapStart, overlapEnd;
          int procs=info->communicator().rank();
          int n = UNKNOWNS/procs; // number of unknowns per process
          int bigger = UNKNOWNS%procs; // number of process with n+1 unknows

          // Compute owner region
          if(rank<bigger) {
            start = rank*(n+1);
            end   = (rank+1)*(n+1);
          }else{
            start = bigger + rank * n;
            end   = bigger + (rank + 1) * n;
          }

          // Compute overlap region
          if(start>0)
            overlapStart = start - 1;
          else
            overlapStart = start;

          if(end<UNKNOWNS)
            overlapEnd = end + 1;
          else
            overlapEnd = end;

          assert((UNKNOWNS)*(overlapEnd-overlapStart)==aggregatesMap->noVertices());
          for(int j=0; j< UNKNOWNS; ++j)
            for(int i=0; i < UNKNOWNS; ++i)
            {
              if(i>=overlapStart && i<overlapEnd)
              {
                int no = (j/2)*((UNKNOWNS)/2)+i/2;
                (*aggregatesMap)[j*(overlapEnd-overlapStart)+i-overlapStart]=no;
              }
            }
        }
#endif
        if(criterion.debugLevel()>1 && info->communicator().rank()==0)
          std::cout<<"aggregating finished."<<std::endl;

        BIGINT gnoAggregates=noAggregates;
        gnoAggregates = info->communicator().sum(gnoAggregates);
        double dgnoAggregates = gnoAggregates.todouble();
#ifdef TEST_AGGLO
        BIGINT gnoAggregates=((UNKNOWNS)/2)*((UNKNOWNS)/2);
#endif

        if(criterion.debugLevel()>2 && rank==0)
          std::cout << "Building "<<dgnoAggregates<<" aggregates took "<<watch.elapsed()<<" seconds."<<std::endl;

        if(dgnoAggregates==0 || dunknowns/dgnoAggregates<criterion.minCoarsenRate())
        {
          if(rank==0)
          {
            if(dgnoAggregates>0)
              std::cerr << "Stopped coarsening because of rate breakdown "<<dunknowns<<"/"<<dgnoAggregates
                        <<"="<<dunknowns/dgnoAggregates<<"<"
                        <<criterion.minCoarsenRate()<<std::endl;
            else
              std::cerr<< "Could not build any aggregates. Probably no connected nodes."<<std::endl;
          }
          aggregatesMap->free();
          delete aggregatesMap;
          aggregatesMaps_.pop_back();

          if(criterion.accumulate() && mlevel.isRedistributed() && info->communicator().size()>1) {
            // coarse level matrix was already redistributed, but to more than 1 process
            // Therefore need to delete the redistribution. Further down it will
            // then be redistributed to 1 process
            delete &(mlevel.getRedistributed().getmat());
            mlevel.deleteRedistributed();
            delete &(infoLevel.getRedistributed());
            infoLevel.deleteRedistributed();
            redistributes_.back().resetSetup();
          }

          break;
        }
        unknowns =  noAggregates;
        dunknowns = dgnoAggregates;

        CommunicationArgs commargs(info->communicator(),info->category());
        parallelInformation_.addCoarser(commargs);

        ++infoLevel; // parallel information on coarse level

        typename PropertyMapTypeSelector<VertexVisitedTag,PropertiesGraph>::Type visitedMap =
          get(VertexVisitedTag(), *(std::get<1>(graphs)));

        watch.reset();
        int aggregates = IndicesCoarsener<ParallelInformation,OverlapFlags>
                         ::coarsen(*info,
                                   *(std::get<1>(graphs)),
                                   visitedMap,
                                   *aggregatesMap,
                                   *infoLevel,
                                   noAggregates);
        GraphCreator::free(graphs);

        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Coarsening of index sets took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        watch.reset();

        infoLevel->buildGlobalLookup(aggregates);
        AggregatesPublisher<Vertex,OverlapFlags,ParallelInformation>::publish(*aggregatesMap,
                                                                              *info,
                                                                              infoLevel->globalLookup());


        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Communicating global aggregate numbers took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        watch.reset();
        std::vector<bool>& visited=excluded;

        typedef std::vector<bool>::iterator Iterator;
        typedef IteratorPropertyMap<Iterator, IdentityMap> VisitedMap2;
        Iterator end = visited.end();
        for(Iterator iter= visited.begin(); iter != end; ++iter)
          *iter=false;

        VisitedMap2 visitedMap2(visited.begin(), Dune::IdentityMap());

        std::shared_ptr<typename MatrixOperator::matrix_type>
          coarseMatrix(productBuilder.build(*(std::get<0>(graphs)), visitedMap2,
                                            *info,
                                            *aggregatesMap,
                                            aggregates,
                                            OverlapFlags()));
        dverb<<"Building of sparsity pattern took "<<watch.elapsed()<<std::endl;
        watch.reset();
        info->freeGlobalLookup();

        delete std::get<0>(graphs);
        productBuilder.calculate(matrix->getmat(), *aggregatesMap, *coarseMatrix, *infoLevel, OverlapFlags());

        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Calculation entries of Galerkin product took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        BIGINT nonzeros = countNonZeros(*coarseMatrix);
        allnonzeros = allnonzeros + infoLevel->communicator().sum(nonzeros);
        MatrixArgs args(coarseMatrix, *infoLevel);

        matrices_.addCoarser(args);
        redistributes_.push_back(RedistributeInfoType());
      } // end level loop


      infoLevel->freeGlobalLookup();

      built_=true;
      AggregatesMap* aggregatesMap=new AggregatesMap(0);
      aggregatesMaps_.push_back(aggregatesMap);

      if(criterion.debugLevel()>0) {
        if(level==criterion.maxLevel()) {
          BIGINT unknownsLevel = mlevel->getmat().N();
          unknownsLevel = infoLevel->communicator().sum(unknownsLevel);
          double dunknownsLevel = unknownsLevel.todouble();
          if(rank==0 && criterion.debugLevel()>1) {
            std::cout<<"Level "<<level<<" has "<<dunknownsLevel<<" unknowns, "<<dunknownsLevel/infoLevel->communicator().size()
                     <<" unknowns per proc (procs="<<infoLevel->communicator().size()<<")"<<std::endl;
          }
        }
      }

      if(criterion.accumulate() && !redistributes_.back().isSetup() &&
         infoLevel->communicator().size()>1) {
#if HAVE_MPI && !HAVE_PARMETIS
        if(criterion.accumulate()==successiveAccu &&
           infoLevel->communicator().rank()==0)
          std::cerr<<"Successive accumulation of data on coarse levels only works with ParMETIS installed."
                   <<"  Fell back to accumulation to one domain on coarsest level"<<std::endl;
#endif

        // accumulate to fewer processors
        std::shared_ptr<Matrix> redistMat = std::make_shared<Matrix>();
        std::shared_ptr<ParallelInformation> redistComm;
        int nodomains = 1;

        repartitionAndDistributeMatrix(mlevel->getmat(), redistMat, *infoLevel,
                                       redistComm, redistributes_.back(), nodomains,criterion);
        MatrixArgs args(redistMat, *redistComm);
        BIGINT unknownsRedist = redistMat->N();
        unknownsRedist = infoLevel->communicator().sum(unknownsRedist);

        if(redistComm->communicator().rank()==0 && criterion.debugLevel()>1) {
          double dunknownsRedist = unknownsRedist.todouble();
          std::cout<<"Level "<<level<<" redistributed has "<<dunknownsRedist<<" unknowns, "<<dunknownsRedist/redistComm->communicator().size()
                   <<" unknowns per proc (procs="<<redistComm->communicator().size()<<")"<<std::endl;
        }
        mlevel.addRedistributed(ConstructionTraits<MatrixOperator>::construct(args));
        infoLevel.addRedistributed(redistComm);
        infoLevel->freeGlobalLookup();
      }

      int levels = matrices_.levels();
      maxlevels_ = parallelInformation_.finest()->communicator().max(levels);
      assert(matrices_.levels()==redistributes_.size());
      if(hasCoarsest() && rank==0 && criterion.debugLevel()>1)
        std::cout<<"operator complexity: "<<allnonzeros.todouble()/finenonzeros.todouble()<<std::endl;

    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::ParallelMatrixHierarchy&
    MatrixHierarchy<M,IS,A>::matrices() const
    {
      return matrices_;
    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::ParallelInformationHierarchy&
    MatrixHierarchy<M,IS,A>::parallelInformation() const
    {
      return parallelInformation_;
    }

    template<class M, class IS, class A>
    void MatrixHierarchy<M,IS,A>::getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data) const
    {
      int levels=aggregatesMaps().size();
      int maxlevels=parallelInformation_.finest()->communicator().max(levels);
      std::size_t size=(*(aggregatesMaps().begin()))->noVertices();
      // We need an auxiliary vector for the consecutive prolongation.
      std::vector<std::size_t> tmp;
      std::vector<std::size_t> *coarse, *fine;

      // make sure the allocated space suffices.
      tmp.reserve(size);
      data.reserve(size);

      // Correctly assign coarse and fine for the first prolongation such that
      // we end up in data in the end.
      if(levels%2==0) {
        coarse=&tmp;
        fine=&data;
      }else{
        coarse=&data;
        fine=&tmp;
      }

      // Number the unknowns on the coarsest level consecutively for each process.
      if(levels==maxlevels) {
        const AggregatesMap& map = *(*(++aggregatesMaps().rbegin()));
        std::size_t m=0;

        for(typename AggregatesMap::const_iterator iter = map.begin(); iter != map.end(); ++iter)
          if(*iter< AggregatesMap::ISOLATED)
            m=std::max(*iter,m);

        coarse->resize(m+1);
        std::size_t i=0;
        srand((unsigned)std::clock());
        std::set<size_t> used;
        for(typename std::vector<std::size_t>::iterator iter=coarse->begin(); iter != coarse->end();
            ++iter, ++i)
        {
          std::pair<std::set<std::size_t>::iterator,bool> ibpair
            = used.insert(static_cast<std::size_t>((((double)rand())/(RAND_MAX+1.0)))*coarse->size());

          while(!ibpair.second)
            ibpair = used.insert(static_cast<std::size_t>((((double)rand())/(RAND_MAX+1.0))*coarse->size()));
          *iter=*(ibpair.first);
        }
      }

      typename ParallelInformationHierarchy::Iterator pinfo = parallelInformation().coarsest();
      --pinfo;

      // Now consecutively project the numbers to the finest level.
      for(typename AggregatesMapList::const_reverse_iterator aggregates=++aggregatesMaps().rbegin();
          aggregates != aggregatesMaps().rend(); ++aggregates,--levels) {

        fine->resize((*aggregates)->noVertices());
        fine->assign(fine->size(), 0);
        Transfer<typename AggregatesMap::AggregateDescriptor, std::vector<std::size_t>, ParallelInformation>
        ::prolongateVector(*(*aggregates), *coarse, *fine, static_cast<std::size_t>(1), *pinfo);
        --pinfo;
        std::swap(coarse, fine);
      }

      // Assertion to check that we really projected to data on the last step.
      assert(coarse==&data);
    }

    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::AggregatesMapList&
    MatrixHierarchy<M,IS,A>::aggregatesMaps() const
    {
      return aggregatesMaps_;
    }
    template<class M, class IS, class A>
    const typename MatrixHierarchy<M,IS,A>::RedistributeInfoList&
    MatrixHierarchy<M,IS,A>::redistributeInformation() const
    {
      return redistributes_;
    }

    template<class M, class IS, class A>
    MatrixHierarchy<M,IS,A>::~MatrixHierarchy()
    {
      typedef typename AggregatesMapList::reverse_iterator AggregatesMapIterator;
      typedef typename ParallelMatrixHierarchy::Iterator Iterator;
      typedef typename ParallelInformationHierarchy::Iterator InfoIterator;

      AggregatesMapIterator amap = aggregatesMaps_.rbegin();
      InfoIterator info = parallelInformation_.coarsest();
      for(Iterator level=matrices_.coarsest(), finest=matrices_.finest(); level != finest;  --level, --info, ++amap) {
        (*amap)->free();
        delete *amap;
      }
      delete *amap;
    }

    template<class M, class IS, class A>
    template<class V, class BA, class TA>
    void MatrixHierarchy<M,IS,A>::coarsenVector(Hierarchy<BlockVector<V,BA>, TA>& hierarchy) const
    {
      assert(hierarchy.levels()==1);
      typedef typename ParallelMatrixHierarchy::ConstIterator Iterator;
      typedef typename RedistributeInfoList::const_iterator RIter;
      RIter redist = redistributes_.begin();

      Iterator matrix = matrices_.finest(), coarsest = matrices_.coarsest();
      int level=0;
      if(redist->isSetup())
        hierarchy.addRedistributedOnCoarsest(matrix.getRedistributed().getmat().N());
      Dune::dvverb<<"Level "<<level<<" has "<<matrices_.finest()->getmat().N()<<" unknowns!"<<std::endl;

      while(matrix != coarsest) {
        ++matrix; ++level; ++redist;
        Dune::dvverb<<"Level "<<level<<" has "<<matrix->getmat().N()<<" unknowns!"<<std::endl;

        hierarchy.addCoarser(matrix->getmat().N());
        if(redist->isSetup())
          hierarchy.addRedistributedOnCoarsest(matrix.getRedistributed().getmat().N());

      }

    }

    template<class M, class IS, class A>
    template<class S, class TA>
    void MatrixHierarchy<M,IS,A>::coarsenSmoother(Hierarchy<S,TA>& smoothers,
                                                  const typename SmootherTraits<S>::Arguments& sargs) const
    {
      assert(smoothers.levels()==0);
      typedef typename ParallelMatrixHierarchy::ConstIterator MatrixIterator;
      typedef typename ParallelInformationHierarchy::ConstIterator PinfoIterator;
      typedef typename AggregatesMapList::const_iterator AggregatesIterator;

      typename ConstructionTraits<S>::Arguments cargs;
      cargs.setArgs(sargs);
      PinfoIterator pinfo = parallelInformation_.finest();
      AggregatesIterator aggregates = aggregatesMaps_.begin();
      int level=0;
      for(MatrixIterator matrix = matrices_.finest(), coarsest = matrices_.coarsest();
          matrix != coarsest; ++matrix, ++pinfo, ++aggregates, ++level) {
        cargs.setMatrix(matrix->getmat(), **aggregates);
        cargs.setComm(*pinfo);
        smoothers.addCoarser(cargs);
      }
      if(maxlevels()>levels()) {
        // This is not the globally coarsest level and therefore smoothing is needed
        cargs.setMatrix(matrices_.coarsest()->getmat(), **aggregates);
        cargs.setComm(*pinfo);
        smoothers.addCoarser(cargs);
        ++level;
      }
    }

    template<class M, class IS, class A>
    template<class F>
    void MatrixHierarchy<M,IS,A>::recalculateGalerkin(const F& copyFlags)
    {
      typedef typename AggregatesMapList::iterator AggregatesMapIterator;
      typedef typename ParallelMatrixHierarchy::Iterator Iterator;
      typedef typename ParallelInformationHierarchy::Iterator InfoIterator;

      AggregatesMapIterator amap = aggregatesMaps_.begin();
      BaseGalerkinProduct productBuilder;
      InfoIterator info = parallelInformation_.finest();
      typename RedistributeInfoList::iterator riIter = redistributes_.begin();
      Iterator level = matrices_.finest(), coarsest=matrices_.coarsest();
      if(level.isRedistributed()) {
        info->buildGlobalLookup(level->getmat().N());
        redistributeMatrixEntries(const_cast<Matrix&>(level->getmat()),
                                  const_cast<Matrix&>(level.getRedistributed().getmat()),
                                  *info,info.getRedistributed(), *riIter);
        info->freeGlobalLookup();
      }

      for(; level!=coarsest; ++amap) {
        const Matrix& fine = (level.isRedistributed() ? level.getRedistributed() : *level).getmat();
        ++level;
        ++info;
        ++riIter;
        productBuilder.calculate(fine, *(*amap), const_cast<Matrix&>(level->getmat()), *info, copyFlags);
        if(level.isRedistributed()) {
          info->buildGlobalLookup(level->getmat().N());
          redistributeMatrixEntries(const_cast<Matrix&>(level->getmat()),
                                    const_cast<Matrix&>(level.getRedistributed().getmat()), *info,
                                    info.getRedistributed(), *riIter);
          info->freeGlobalLookup();
        }
      }
    }

    template<class M, class IS, class A>
    std::size_t MatrixHierarchy<M,IS,A>::levels() const
    {
      return matrices_.levels();
    }

    template<class M, class IS, class A>
    std::size_t MatrixHierarchy<M,IS,A>::maxlevels() const
    {
      return maxlevels_;
    }

    template<class M, class IS, class A>
    bool MatrixHierarchy<M,IS,A>::hasCoarsest() const
    {
      return levels()==maxlevels() &&
             (!matrices_.coarsest().isRedistributed() ||matrices_.coarsest()->getmat().N()>0);
    }

    template<class M, class IS, class A>
    bool MatrixHierarchy<M,IS,A>::isBuilt() const
    {
      return built_;
    }

    /** @} */
  } // namespace Amg
} // namespace Dune

#endif // end DUNE_AMG_MATRIXHIERARCHY_HH
