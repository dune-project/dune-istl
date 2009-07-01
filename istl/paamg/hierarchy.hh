// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_AMGHIERARCHY_HH
#define DUNE_AMGHIERARCHY_HH

#include <list>
#include <memory>
#include <limits>
#include <algorithm>
#include "pmatrix.hh"
#include "aggregates.hh"
#include "graph.hh"
#include "galerkin.hh"
#include "renumberer.hh"
#include "graphcreator.hh"
#include <dune/common/stdstreams.hh>
#include <dune/common/timer.hh>
#include <dune/common/tuples.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/indexset.hh>
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
    /**
     * @brief A hierarchy of coantainers (e.g. matrices or vectors)
     *
     * Because sometimes a redistribution of the parallel data might be
     * advisable one can add redistributed version of the container at
     * each level.
     */
    template<typename T, typename A=std::allocator<T> >
    class Hierarchy
    {
    public:
      /**
       * @brief The type of the container we store.
       */
      typedef T MemberType;

      template<typename T1, typename T2>
      class LevelIterator;

    private:
      /**
       * @brief An element in the hierarchy.
       */
      struct Element
      {
        friend class LevelIterator<Hierarchy<T,A>, T>;
        friend class LevelIterator<const Hierarchy<T,A>, const T>;

        /** @brief The next coarser element in the list. */
        Element* coarser_;

        /** @brief The next finer element in the list. */
        Element* finer_;

        /** @brief Pointer to the element. */
        MemberType* element_;

        /** @brief The redistributed version of the element. */
        MemberType* redistributed_;
      };
    public:
      //       enum{
      //        /**
      //         * @brief If true only the method addCoarser will be usable
      //         * otherwise only the method addFiner will be usable.
      //         */
      //        coarsen = b
      //          };

      /**
       * @brief The allocator to use for the list elements.
       */
      typedef typename A::template rebind<Element>::other Allocator;

      typedef typename ConstructionTraits<T>::Arguments Arguments;

      /**
       * @brief Construct a new hierarchy.
       * @param first The first element in the hierarchy.
       */
      Hierarchy(MemberType& first);

      /**
       * @brief Construct a new empty hierarchy.
       */
      Hierarchy();

      /**
       * @brief Add an element on a coarser level.
       * @param args The arguments needed for the construction.
       */
      void addCoarser(Arguments& args);


      /**
       * @brief Add an element on a finer level.
       * @param args The arguments needed for the construction.
       */
      void addFiner(Arguments& args);

      /**
       * @brief Iterator over the levels in the hierarchy.
       *
       * operator++() moves to the next coarser level in the hierarchy.
       * while operator--() moves to the next finer level in the hierarchy.
       */
      template<class C, class T1>
      class LevelIterator
        : public BidirectionalIteratorFacade<LevelIterator<C,T1>,T1,T1&>
      {
        friend class LevelIterator<typename remove_const<C>::type,
            typename remove_const<T1>::type >;
        friend class LevelIterator<const typename remove_const<C>::type,
            const typename remove_const<T1>::type >;

      public:
        /** @brief Constructor. */
        LevelIterator()
          : element_(0)
        {}

        LevelIterator(Element* element)
          : element_(element)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<typename remove_const<C>::type,
                          typename remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<const typename remove_const<C>::type,
                          const typename remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<typename remove_const<C>::type,
                        typename remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<const typename remove_const<C>::type,
                        const typename remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /** @brief Dereference the iterator. */
        T1& dereference() const
        {
          return *(element_->element_);
        }

        /** @brief Move to the next coarser level */
        void increment()
        {
          element_ = element_->coarser_;
        }

        /** @brief Move to the next fine level */
        void decrement()
        {
          element_ = element_->finer_;
        }

        /**
         * @brief Check whether there was a redistribution at the current level.
         * @return True if there is a redistributed version of the conatainer at the current level.
         */
        bool isRedistributed() const
        {
          return element_->redistributed_;
        }

        /**
         * @brief Get the redistributed container.
         * @return The redistributed container.
         */
        T1& getRedistributed() const
        {
          assert(element_->redistributed_);
          return *element_->redistributed_;
        }

      private:
        Element* element_;
      };

      /** @brief Type of the mutable iterator. */
      typedef LevelIterator<Hierarchy<T,A>,T> Iterator;

      /** @brief Type of the const iterator. */
      typedef LevelIterator<const Hierarchy<T,A>, const T> ConstIterator;

      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      Iterator finest();

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      Iterator coarsest();


      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      ConstIterator finest() const;

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      ConstIterator coarsest() const;

      /**
       * @brief Get the number of levels in the hierarchy.
       * @return The number of levels.
       */
      int levels() const;

      /** @brief Destructor. */
      ~Hierarchy();

    private:
      /** @brief The finest element in the hierarchy. */
      Element* finest_;
      /** @brief The coarsest element in the hierarchy. */
      Element* coarsest_;
      /** @brief Whether the first element was not allocated by us. */
      Element* nonAllocated_;
      /** @brief The allocator for the list elements. */
      Allocator allocator_;
      /** @brief The number of levels in the hierarchy. */
      int levels_;
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
      typedef typename Allocator::template rebind<AggregatesMap*>::other AAllocator;

      /** @brief The type of the aggregates maps list. */
      typedef std::list<AggregatesMap*,AAllocator> AggregatesMapList;

      /**
       * @brief Constructor
       * @param fineMatrix The matrix to coarsen.
       * @param pinfo The information about the parallel data decomposition at the first level.
       */
      MatrixHierarchy(const MatrixOperator& fineMatrix,
                      const ParallelInformation& pinfo=ParallelInformation());


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
       * this will recalculate all coarser level without starting the expensive
       * aggregation process all over again.
       */
      template<class F>
      void recalculateGalerkin(const F& copyFlags);

      /**
       * @brief Coarsen the vector hierarchy according to the matrix hierarchy.
       * @param hierarchy The vector hierarchy to coarsen.
       */
      template<class V, class TA>
      void coarsenVector(Hierarchy<BlockVector<V,TA> >& hierarchy) const;

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
      int levels() const;

      /**
       * @brief Whether the hierarchy wis built.
       * @return true if the ::coarsen method was called.
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

      typename MatrixOperator::field_type getProlongationDampingFactor() const
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
      void getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data);

    private:
      typedef typename ConstructionTraits<MatrixOperator>::Arguments MatrixArgs;
      /** @brief The list of aggregates maps. */
      AggregatesMapList aggregatesMaps_;
      /** @brief The hierarchy of parallel matrices. */
      ParallelMatrixHierarchy matrices_;
      /** @brief The hierarchy of the parallel information. */
      ParallelInformationHierarchy parallelInformation_;

      /** @brief Whether the hierarchy was built. */
      bool built_;

      typename MatrixOperator::field_type prolongDamp_;

      /**
       * @brief functor to print matrix statistics.
       */
      template<class Matrix, bool print>
      struct MatrixStats
      {

        /**
         * @brief Print matrix statistics.
         */
        static void stats(const Matrix& matrix)
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
       * This might be e.g. SymmetricDependency or UnSymmetricCriterion.
       */
      typedef T DependencyCriterion;

      /**
       * @brief Set the maximum number of levels allowed in the hierarchy.
       */
      void setMaxLevel(int l)
      {
        maxLevel_ = l;
      }
      /**
       * @brief Get the maximum number of levels allowed in the hierarchy.
       */
      int maxLevel() const
      {
        return maxLevel_;
      }

      /**
       * @brief Set the maximum number of unknowns allowed on the coarsest level.
       */
      void setCoarsenTarget(int nodes)
      {
        coarsenTarget_ = nodes;
      }

      /**
       * @brief Get the maximum number of unknowns allowed on the coarsest level.
       */
      int coarsenTarget() const
      {
        return coarsenTarget_;
      }

      /**
       * @brief Set the minimum coarsening rate to be achieved in each coarsening.
       *
       * The default value is 1.2
       */
      void setMinCoarsenRate(double rate)
      {
        minCoarsenRate_ = rate;
      }

      /**
       * @brief Get the minimum coarsening rate to be achieved.
       */
      double minCoarsenRate() const
      {
        return minCoarsenRate_;
      }

      /**
       * @brief Whether the data should be accumulated on fewer processes on coarser levels.
       */
      bool accumulate() const
      {
        return false;
      }

      /**
       * @brief Set the damping factor for the prolongation.
       *
       * @param d The new damping factor.
       */
      void setProlongationDampingFactor(double d)
      {
        dampingFactor_ = d;
      }

      /**
       * @brief Get the damping factor for the prolongation.
       *
       * @return d The damping factor.
       */
      double getProlongationDampingFactor() const
      {
        return dampingFactor_;
      }
      /**
       * @brief Constructor
       * @param maxLevel The macimum number of levels allowed in the matric hierarchy (default: 100).
       * @param coarsenTarget If the number of nodes in the matrix is below this threshold the
       * coarsening will stop (default: 1000).
       * @param minCoarsenRate If the coarsening rate falls below this threshold the
       * coarsening will stop (default: 1.2)
       * @param prolongDamp The damping factor to apply to the prolongated update (default: 1.6)
       */
      CoarsenCriterion(int maxLevel=100, int coarsenTarget=1000, double minCoarsenRate=1.2,
                       double prolongDamp=1.6)
        : T(), maxLevel_(maxLevel), coarsenTarget_(coarsenTarget), minCoarsenRate_(minCoarsenRate),
          dampingFactor_(prolongDamp)
      {}

    private:
      /**
       * @brief The maximum number of levels allowed in the hierarchy.
       */
      int maxLevel_;
      /**
       * @brief The maximum number of unknowns allowed on the coarsest level.
       */
      int coarsenTarget_;
      /**
       * @brief The minimum coarsening rate to be achieved.
       */
      double minCoarsenRate_;
      /**
       * @brief The damping factor to apply to the prologated correction.
       */
      double dampingFactor_;
    };


    template<class M, class IS, class A>
    MatrixHierarchy<M,IS,A>::MatrixHierarchy(const MatrixOperator& fineOperator,
                                             const ParallelInformation& pinfo)
      : matrices_(const_cast<MatrixOperator&>(fineOperator)),
        parallelInformation_(const_cast<ParallelInformation&>(pinfo))
    {
      dune_static_assert((static_cast<int>(MatrixOperator::category) == static_cast<int>(SolverCategory::sequential) ||
                          static_cast<int>(MatrixOperator::category) == static_cast<int>(SolverCategory::overlapping)),
                         "MatrixOperator must be of category sequential or overlapping");
      dune_static_assert((static_cast<int>(MatrixOperator::category) == static_cast<int>(ParallelInformation::category)),
                         "MatrixOperator and ParallelInformation must belong to the same category!");
    }

    template<class M, class IS, class A>
    template<typename O, typename T>
    void MatrixHierarchy<M,IS,A>::build(const T& criterion)
    {

      prolongDamp_ = criterion.getProlongationDampingFactor();
      typedef O OverlapFlags;
      typedef typename ParallelMatrixHierarchy::Iterator MatIterator;
      typedef typename ParallelInformationHierarchy::Iterator PInfoIterator;

      GalerkinProduct<ParallelInformation> productBuilder;
      MatIterator mlevel = matrices_.finest();
      MatrixStats<typename M::matrix_type,MINIMAL_DEBUG_LEVEL<=INFO_DEBUG_LEVEL>::stats(mlevel->getmat());

      PInfoIterator infoLevel = parallelInformation_.finest();


      int procs = infoLevel->communicator().size();
      int level = 0;
      int rank = 0;
      int unknowns = mlevel->getmat().N();;

      unknowns = infoLevel->communicator().sum(unknowns);
      infoLevel->buildGlobalLookup(mlevel->getmat().N());

      for(; level < criterion.maxLevel(); ++level, ++mlevel) {

        rank = infoLevel->communicator().rank();
        if(rank==0 && criterion.debugLevel()>1)
          std::cout<<"Level "<<level<<" has "<<unknowns<<" unknowns, "<<unknowns/infoLevel->communicator().size()<<" unknowns per proc"<<std::endl;

        if(unknowns <= criterion.coarsenTarget())
          // No further coarsening needed
          break;

        typedef PropertiesGraphCreator<MatrixOperator> GraphCreator;
        typedef typename GraphCreator::PropertiesGraph PropertiesGraph;
        typedef typename GraphCreator::MatrixGraph MatrixGraph;
        typedef typename GraphCreator::GraphTuple GraphTuple;

        typedef typename PropertiesGraph::VertexDescriptor Vertex;

        std::vector<bool> excluded(mlevel->getmat().N(), false);

        GraphTuple graphs = GraphCreator::create(*mlevel, excluded, *infoLevel, OverlapFlags());

        AggregatesMap* aggregatesMap=new AggregatesMap(get<1>(graphs)->maxVertex());

        aggregatesMaps_.push_back(aggregatesMap);

        Timer watch;
        watch.reset();
        int noAggregates, isoAggregates, oneAggregates;

        tie(noAggregates, isoAggregates, oneAggregates) =
          aggregatesMap->buildAggregates(mlevel->getmat(), *(get<1>(graphs)), criterion);

#ifdef TEST_AGGLO
        {
          // calculate size of local matrix in the distributed direction
          int start, end, overlapStart, overlapEnd;
          int n = UNKNOWNS/procs; // number of unknowns per process
          int bigger = UNKNOWNS%procs; // number of process with n+1 unknows
          int procs=infoLevel->communicator().rank();

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

          int noKnown = overlapEnd-overlapStart;
          int offset = start-overlapStart;
          int starti = 1; //(start-overlapStart==0)?1:0;
          int endi = (overlapEnd-end==0) ? end-start-1 : end-start;

          for(int j=1; j< UNKNOWNS-1; ++j)
            for(int i=starti; i < endi; ++i)
              (*aggregatesMap)[j*(overlapEnd-overlapStart)+i+offset]=((j-1)/2)*(endi-starti)/2+((i-starti)/2);
          noAggregates=((UNKNOWNS-2)/2)*(endi-starti)/2;
        }
#endif
        noAggregates = infoLevel->communicator().sum(noAggregates);

        if(criterion.debugLevel()>2 && rank==0)
          std::cout << "Building "<<noAggregates<<" aggregates took "<<watch.elapsed()<<" seconds."<<std::endl;

        if(!noAggregates || unknowns/noAggregates<criterion.minCoarsenRate())
        {
          if(procs>1 && criterion.accumulate())
            DUNE_THROW(NotImplemented, "Accumulation to fewer processes not yet implemented!");
          else{
            if(rank==0)
            {
              if(noAggregates)
                std::cerr << "Stopped coarsening because of rate breakdown "<<unknowns/noAggregates<<"<"
                          <<criterion.minCoarsenRate()<<std::endl;
              else
                std::cerr<< "Could not build any aggregates. Probably no connected nodes."<<std::endl;
            }
            aggregatesMap->free();
            delete aggregatesMap;
            aggregatesMaps_.pop_back();
            break;
          }
        }
        unknowns =  noAggregates;

        if(noAggregates < criterion.coarsenTarget() && procs>1 && criterion.accumulate()) {
          DUNE_THROW(NotImplemented, "Accumulation to fewer processes not yet implemented!");
        }

        parallelInformation_.addCoarser(infoLevel->communicator());

        PInfoIterator fineInfo = infoLevel++;

        typename PropertyMapTypeSelector<VertexVisitedTag,PropertiesGraph>::Type visitedMap =
          get(VertexVisitedTag(), *(get<1>(graphs)));

        watch.reset();
        int aggregates = IndicesCoarsener<ParallelInformation,OverlapFlags>
                         ::coarsen(*fineInfo,
                                   *(get<1>(graphs)),
                                   visitedMap,
                                   *aggregatesMap,
                                   *infoLevel);

        GraphCreator::free(graphs);

        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Coarsening of index sets took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        watch.reset();

        infoLevel->buildGlobalLookup(aggregates);
        AggregatesPublisher<Vertex,OverlapFlags,ParallelInformation>::publish(*aggregatesMap,
                                                                              *fineInfo,
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

        typename MatrixOperator::matrix_type* coarseMatrix;

        coarseMatrix = productBuilder.build(mlevel->getmat(), *(get<0>(graphs)), visitedMap2,
                                            *fineInfo,
                                            *aggregatesMap,
                                            aggregates,
                                            OverlapFlags());

        fineInfo->freeGlobalLookup();

        delete get<0>(graphs);
        productBuilder.calculate(mlevel->getmat(), *aggregatesMap, *coarseMatrix, *infoLevel, OverlapFlags());

        if(criterion.debugLevel()>2) {
          if(rank==0)
            std::cout<<"Calculation of Galerkin product took "<<watch.elapsed()<<" seconds."<<std::endl;
        }

        MatrixArgs args(*coarseMatrix, *infoLevel);

        matrices_.addCoarser(args);
      }

      infoLevel->freeGlobalLookup();

      built_=true;
      AggregatesMap* aggregatesMap=new AggregatesMap(0);
      aggregatesMaps_.push_back(aggregatesMap);

      if(criterion.debugLevel()>0) {
        if(level==criterion.maxLevel()) {
          int unknowns = mlevel->getmat().N();
          unknowns = infoLevel->communicator().sum(unknowns);
          if(rank==0 && criterion.debugLevel()>1)
            std::cout<<"Level "<<level<<" has "<<unknowns<<" unknowns, "<<unknowns/infoLevel->communicator().size()<<" unknowns per proc"<<std::endl;
        }
      }
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
    void MatrixHierarchy<M,IS,A>::getCoarsestAggregatesOnFinest(std::vector<std::size_t>& data)
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
          m=std::max(*iter,m);

        coarse->resize(m+1);
        std::size_t i=0;

        for(typename std::vector<std::size_t>::iterator iter=coarse->begin(); iter != coarse->end();
            ++iter, ++i)
          *iter=i;
      }

      typename ParallelInformationHierarchy::Iterator pinfo = parallelInformation().coarsest();
      --pinfo;
      int l=levels;

      // Now consecutively project the numbers to the finest level.
      for(typename AggregatesMapList::const_reverse_iterator aggregates=++aggregatesMaps().rbegin();
          aggregates != aggregatesMaps().rend(); ++aggregates,--levels) {

        fine->resize((*aggregates)->noVertices());
        fine->assign(fine->size(), 0);
        Transfer<typename AggregatesMap::AggregateDescriptor, std::vector<std::size_t>, ParallelInformation>
        ::prolongate(*(*aggregates), *coarse, *fine, static_cast<std::size_t>(1), *pinfo);
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
        delete &level->getmat();
      }
      delete *amap;
    }

    template<class M, class IS, class A>
    template<class V, class TA>
    void MatrixHierarchy<M,IS,A>::coarsenVector(Hierarchy<BlockVector<V,TA> >& hierarchy) const
    {
      assert(hierarchy.levels()==1);
      typedef typename ParallelMatrixHierarchy::ConstIterator Iterator;
      Iterator coarsest = matrices_.coarsest();
      int level=0;
      Dune::dvverb<<"Level "<<level<<" has "<<matrices_.finest()->getmat().N()<<" unknows!"<<std::endl;

      for(Iterator matrix = matrices_.finest(); matrix != coarsest;) {
        ++matrix;
        ++level;
        Dune::dvverb<<"Level "<<level<<" has "<<matrix->getmat().N()<<" unknows!"<<std::endl;
        hierarchy.addCoarser(matrix->getmat().N());
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
      for(MatrixIterator matrix = matrices_.finest(), coarsest = matrices_.coarsest();
          matrix != coarsest; ++matrix, ++pinfo, ++aggregates) {
        cargs.setMatrix(matrix->getmat(), **aggregates);
        cargs.setComm(*pinfo);
        smoothers.addCoarser(cargs);
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

      for(Iterator level = matrices_.finest(), coarsest=matrices_.coarsest(); level!=coarsest; ++amap) {
        const Matrix& fine = level->getmat();
        ++level;
        ++info;
        productBuilder.calculate(fine, *(*amap), const_cast<Matrix&>(level->getmat()), *info, copyFlags);

      }
    }

    template<class M, class IS, class A>
    int MatrixHierarchy<M,IS,A>::levels() const
    {
      return matrices_.levels();
    }

    template<class M, class IS, class A>
    bool MatrixHierarchy<M,IS,A>::isBuilt() const
    {
      return built_;
    }

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy()
      : finest_(0), coarsest_(0), nonAllocated_(0), allocator_(), levels_(0)
    {}

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(MemberType& first)
      : allocator_()
    {
      finest_ = allocator_.allocate(1,0);
      finest_->element_ = &first;
      finest_->redistributed_ = 0;
      nonAllocated_ = finest_;
      coarsest_ = finest_;
      coarsest_->coarser_ = coarsest_->finer_ = 0;
      levels_ = 1;
    }

    template<class T, class A>
    Hierarchy<T,A>::~Hierarchy()
    {
      while(coarsest_) {
        Element* current = coarsest_;
        coarsest_ = coarsest_->finer_;
        if(current != nonAllocated_) {
          ConstructionTraits<T>::deconstruct(current->element_);
        }
        allocator_.deallocate(current, 1);
        //coarsest_->coarser_ = 0;
      }
    }

    template<class T, class A>
    int Hierarchy<T,A>::levels() const
    {
      return levels_;
    }

    template<class T, class A>
    void Hierarchy<T,A>::addCoarser(Arguments& args)
    {
      if(!coarsest_) {
        assert(!finest_);
        coarsest_ = allocator_.allocate(1,0);
        coarsest_->element_ = ConstructionTraits<MemberType>::construct(args);
        finest_ = coarsest_;
        coarsest_->finer_ = 0;
      }else{
        coarsest_->coarser_ = allocator_.allocate(1,0);
        coarsest_->coarser_->finer_ = coarsest_;
        coarsest_ = coarsest_->coarser_;
        coarsest_->element_ = ConstructionTraits<MemberType>::construct(args);
      }
      coarsest_->redistributed_ = 0;
      coarsest_->coarser_=0;
      ++levels_;
    }

    template<class T, class A>
    void Hierarchy<T,A>::addFiner(Arguments& args)
    {
      if(!finest_) {
        assert(!coarsest_);
        finest_ = allocator_.allocate(1,0);
        finest_->element = ConstructionTraits<T>::construct(args);
        coarsest_ = finest_;
        coarsest_->coarser_ = coarsest_->finer_ = 0;
      }else{
        finest_->finer_ = allocator_.allocate(1,0);
        finest_->finer_->coarser_ = finest_;
        finest_ = finest_->finer_;
        finest_->finer = 0;
        finest_->element = ConstructionTraits<T>::construct(args);
      }
      ++levels_;
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::finest()
    {
      return Iterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::coarsest()
    {
      return Iterator(coarsest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::finest() const
    {
      return ConstIterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::coarsest() const
    {
      return ConstIterator(coarsest_);
    }
    /** @} */
  } // namespace Amg
} // namespace Dune

#endif
