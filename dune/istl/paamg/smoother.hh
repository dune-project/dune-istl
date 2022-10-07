// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGSMOOTHER_HH
#define DUNE_AMGSMOOTHER_HH

#include <dune/istl/paamg/construction.hh>
#include <dune/istl/paamg/aggregates.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/common/propertymap.hh>
#include <dune/common/ftraits.hh>

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
     * @brief Classes for the generic construction and application
     * of the smoothers.
     */

    /**
     * @brief The default class for the smoother arguments.
     */
    template<class T>
    struct DefaultSmootherArgs
    {
      /**
       * @brief The type of the relaxation factor.
       */
      typedef typename FieldTraits<T>::real_type RelaxationFactor;

      /**
       * @brief The numbe of iterations to perform.
       */
      int iterations;
      /**
       * @brief The relaxation factor to use.
       */
      RelaxationFactor relaxationFactor;

      /**
       * @brief Default constructor.
       */
      DefaultSmootherArgs()
        : iterations(1), relaxationFactor(1.0)
      {}
    };

    /**
     * @brief Traits class for getting the attribute class of a smoother.
     */
    template<class T>
    struct SmootherTraits
    {
      typedef DefaultSmootherArgs<typename T::matrix_type::field_type> Arguments;

    };

    template<class X, class Y>
    struct SmootherTraits<Richardson<X,Y>>
    {
      typedef DefaultSmootherArgs<typename X::field_type> Arguments;

    };

    template<class X, class Y, class C, class T>
    struct SmootherTraits<BlockPreconditioner<X,Y,C,T> >
        : public SmootherTraits<T>
    {};

    template<class C, class T>
    struct SmootherTraits<NonoverlappingBlockPreconditioner<C,T> >
        : public SmootherTraits<T>
    {};

    /**
     * @brief Construction Arguments for the default smoothers
     */
    template<class T>
    class DefaultConstructionArgs
    {
      typedef typename T::matrix_type Matrix;

      typedef typename SmootherTraits<T>::Arguments SmootherArgs;

      typedef Dune::Amg::AggregatesMap<typename MatrixGraph<Matrix>::VertexDescriptor> AggregatesMap;

    public:
      virtual ~DefaultConstructionArgs()
      {}

      void setMatrix(const Matrix& matrix)
      {
        matrix_=&matrix;
      }
      virtual void setMatrix(const Matrix& matrix, [[maybe_unused]] const AggregatesMap& amap)
      {
        setMatrix(matrix);
      }


      const Matrix& getMatrix() const
      {
        return *matrix_;
      }

      void setArgs(const SmootherArgs& args)
      {
        args_=&args;
      }

      template<class T1>
      void setComm([[maybe_unused]] T1& comm)
      {}

      const SequentialInformation& getComm()
      {
        return comm_;
      }

      const SmootherArgs getArgs() const
      {
        return *args_;
      }

    protected:
      const Matrix* matrix_;
    private:
      const SmootherArgs* args_;
      SequentialInformation comm_;
    };

    template<class T>
    struct ConstructionArgs
      : public DefaultConstructionArgs<T>
    {};

    template<class T, class C=SequentialInformation>
    class DefaultParallelConstructionArgs
      : public ConstructionArgs<T>
    {
    public:
      virtual ~DefaultParallelConstructionArgs()
      {}

      void setComm(const C& comm)
      {
        comm_ = &comm;
      }

      const C& getComm() const
      {
        return *comm_;
      }
    private:
      const C* comm_;
    };


    template<class X, class Y>
    class DefaultConstructionArgs<Richardson<X,Y>>
    {
      typedef Richardson<X,Y> T;

      typedef typename SmootherTraits<T>::Arguments SmootherArgs;

    public:
      virtual ~DefaultConstructionArgs()
      {}

      template <class... Args>
      void setMatrix(const Args&...)
      {}

      void setArgs(const SmootherArgs& args)
      {
        args_=&args;
      }

      template<class T1>
      void setComm([[maybe_unused]] T1& comm)
      {}

      const SequentialInformation& getComm()
      {
        return comm_;
      }

      const SmootherArgs getArgs() const
      {
        return *args_;
      }

    private:
      const SmootherArgs* args_;
      SequentialInformation comm_;
    };



    template<class T>
    struct ConstructionTraits;

    /**
     * @brief Policy for the construction of the SeqSSOR smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqSSOR<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqSSOR<M,X,Y,l> > Arguments;

      static inline std::shared_ptr<SeqSSOR<M,X,Y,l>> construct(Arguments& args)
      {
        return std::make_shared<SeqSSOR<M,X,Y,l>>
          (args.getMatrix(), args.getArgs().iterations, args.getArgs().relaxationFactor);
      }
    };


    /**
     * @brief Policy for the construction of the SeqSOR smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqSOR<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqSOR<M,X,Y,l> > Arguments;

      static inline std::shared_ptr<SeqSOR<M,X,Y,l>> construct(Arguments& args)
      {
        return std::make_shared<SeqSOR<M,X,Y,l>>
          (args.getMatrix(), args.getArgs().iterations, args.getArgs().relaxationFactor);
      }
    };


    /**
     * @brief Policy for the construction of the SeqJac smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqJac<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqJac<M,X,Y,l> > Arguments;

      static inline std::shared_ptr<SeqJac<M,X,Y,l>> construct(Arguments& args)
      {
        return std::make_shared<SeqJac<M,X,Y,l>>
          (args.getMatrix(), args.getArgs().iterations, args.getArgs().relaxationFactor);
      }
    };

    /**
     * @brief Policy for the construction of the Richardson smoother
     */
    template<class X, class Y>
    struct ConstructionTraits<Richardson<X,Y> >
    {
      typedef DefaultConstructionArgs<Richardson<X,Y> > Arguments;

      static inline std::shared_ptr<Richardson<X,Y>> construct(Arguments& args)
      {
        return std::make_shared<Richardson<X,Y>>
          (args.getArgs().relaxationFactor);
      }
    };


    template<class M, class X, class Y>
    class ConstructionArgs<SeqILU<M,X,Y> >
      : public DefaultConstructionArgs<SeqILU<M,X,Y> >
    {
    public:
      ConstructionArgs(int n=0)
        : n_(n)
      {}

      void setN(int n)
      {
        n_ = n;
      }

      int getN()
      {
        return n_;
      }

    private:
      int n_;
    };


    /**
     * @brief Policy for the construction of the SeqILU smoother
     */
    template<class M, class X, class Y>
    struct ConstructionTraits<SeqILU<M,X,Y> >
    {
      typedef ConstructionArgs<SeqILU<M,X,Y> > Arguments;

      static inline std::shared_ptr<SeqILU<M,X,Y>> construct(Arguments& args)
      {
        return std::make_shared<SeqILU<M,X,Y>>
          (args.getMatrix(), args.getN(), args.getArgs().relaxationFactor);
      }
    };

    /**
     * @brief Policy for the construction of the ParSSOR smoother
     */
    template<class M, class X, class Y, class C>
    struct ConstructionTraits<ParSSOR<M,X,Y,C> >
    {
      typedef DefaultParallelConstructionArgs<M,C> Arguments;

      static inline std::shared_ptr<ParSSOR<M,X,Y,C>> construct(Arguments& args)
      {
        return std::make_shared<ParSSOR<M,X,Y,C>>
          (args.getMatrix(), args.getArgs().iterations,
           args.getArgs().relaxationFactor, args.getComm());
      }
    };

    template<class X, class Y, class C, class T>
    struct ConstructionTraits<BlockPreconditioner<X,Y,C,T> >
    {
      typedef DefaultParallelConstructionArgs<T,C> Arguments;
      typedef ConstructionTraits<T> SeqConstructionTraits;
      static inline std::shared_ptr<BlockPreconditioner<X,Y,C,T>> construct(Arguments& args)
      {
        auto seqPrec = SeqConstructionTraits::construct(args);
        return std::make_shared<BlockPreconditioner<X,Y,C,T>> (seqPrec, args.getComm());
      }
    };

    template<class C, class T>
    struct ConstructionTraits<NonoverlappingBlockPreconditioner<C,T> >
    {
      typedef DefaultParallelConstructionArgs<T,C> Arguments;
      typedef ConstructionTraits<T> SeqConstructionTraits;
      static inline std::shared_ptr<NonoverlappingBlockPreconditioner<C,T>> construct(Arguments& args)
      {
        auto seqPrec = SeqConstructionTraits::construct(args);
        return std::make_shared<NonoverlappingBlockPreconditioner<C,T>> (seqPrec, args.getComm());
      }
    };

    /**
     * @brief Helper class for applying the smoothers.
     *
     * The goal of this class is to get a symmetric AMG method
     * whenever possible.
     *
     * The specializations for SOR and SeqOverlappingSchwarz in
     * MultiplicativeSchwarzMode will apply
     * the smoother forward when pre and backward when post smoothing.
     */
    template<class T>
    struct SmootherApplier
    {
      typedef T Smoother;
      typedef typename Smoother::range_type Range;
      typedef typename Smoother::domain_type Domain;

      /**
       * @brief apply pre smoothing in forward direction
       *
       * @param smoother The smoother to use.
       * @param d The current defect.
       * @param v handle to store the update in.
       */
      static void preSmooth(Smoother& smoother, Domain& v, const Range& d)
      {
        smoother.apply(v,d);
      }

      /**
       * @brief apply post smoothing in forward direction
       *
       * @param smoother The smoother to use.
       * @param d The current defect.
       * @param v handle to store the update in.
       */
      static void postSmooth(Smoother& smoother, Domain& v, const Range& d)
      {
        smoother.apply(v,d);
      }
    };

    /**
     * @brief Apply pre smoothing on the current level.
     * @param levelContext the iterators of the current level.
     * @param steps The number of smoothing steps to apply.
     */
    template<typename LevelContext>
    void presmooth(LevelContext& levelContext, size_t steps)
    {
        for(std::size_t i=0; i < steps; ++i) {
          *levelContext.lhs=0;
          SmootherApplier<typename LevelContext::SmootherType>
            ::preSmooth(*levelContext.smoother, *levelContext.lhs,
                        *levelContext.rhs);
          // Accumulate update
          *levelContext.update += *levelContext.lhs;

          // update defect
          levelContext.matrix->applyscaleadd(-1, *levelContext.lhs, *levelContext.rhs);
          levelContext.pinfo->project(*levelContext.rhs);
        }
    }

    /**
     * @brief Apply post smoothing on the current level.
     * @param levelContext the iterators of the current level.
     * @param steps The number of smoothing steps to apply.
     */
    template<typename LevelContext>
    void postsmooth(LevelContext& levelContext, size_t steps)
    {
        for(std::size_t i=0; i < steps; ++i) {
          // update defect
          levelContext.matrix->applyscaleadd(-1, *levelContext.lhs,
                                             *levelContext.rhs);
          *levelContext.lhs=0;
          levelContext.pinfo->project(*levelContext.rhs);
          SmootherApplier<typename LevelContext::SmootherType>
            ::postSmooth(*levelContext.smoother, *levelContext.lhs, *levelContext.rhs);
          // Accumulate update
          *levelContext.update += *levelContext.lhs;
        }
    }

    template<class M, class X, class Y, int l>
    struct SmootherApplier<SeqSOR<M,X,Y,l> >
    {
      typedef SeqSOR<M,X,Y,l> Smoother;
      typedef typename Smoother::range_type Range;
      typedef typename Smoother::domain_type Domain;

      static void preSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<true>(v,d);
      }


      static void postSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<false>(v,d);
      }
    };

    template<class M, class X, class Y, class C, int l>
    struct SmootherApplier<BlockPreconditioner<X,Y,C,SeqSOR<M,X,Y,l> > >
    {
      typedef BlockPreconditioner<X,Y,C,SeqSOR<M,X,Y,l> > Smoother;
      typedef typename Smoother::range_type Range;
      typedef typename Smoother::domain_type Domain;

      static void preSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<true>(v,d);
      }


      static void postSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<false>(v,d);
      }
    };

    template<class M, class X, class Y, class C, int l>
    struct SmootherApplier<NonoverlappingBlockPreconditioner<C,SeqSOR<M,X,Y,l> > >
    {
      typedef NonoverlappingBlockPreconditioner<C,SeqSOR<M,X,Y,l> > Smoother;
      typedef typename Smoother::range_type Range;
      typedef typename Smoother::domain_type Domain;

      static void preSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<true>(v,d);
      }


      static void postSmooth(Smoother& smoother, Domain& v, Range& d)
      {
        smoother.template apply<false>(v,d);
      }
    };

  } // end namespace Amg

  // forward declarations
  template<class M, class X, class MO, class MS, class A>
  class SeqOverlappingSchwarz;

  struct MultiplicativeSchwarzMode;

  namespace Amg
  {
    template<class M, class X, class MS, class TA>
    struct SmootherApplier<SeqOverlappingSchwarz<M,X,MultiplicativeSchwarzMode,
            MS,TA> >
    {
      typedef SeqOverlappingSchwarz<M,X,MultiplicativeSchwarzMode,MS,TA> Smoother;
      typedef typename Smoother::range_type Range;
      typedef typename Smoother::domain_type Domain;

      static void preSmooth(Smoother& smoother, Domain& v, const Range& d)
      {
        smoother.template apply<true>(v,d);
      }


      static void postSmooth(Smoother& smoother, Domain& v, const Range& d)
      {
        smoother.template apply<false>(v,d);

      }
    };

    //    template<class M, class X, class TM, class TA>
    //    class SeqOverlappingSchwarz;

    template<class T>
    struct SeqOverlappingSchwarzSmootherArgs
      : public DefaultSmootherArgs<T>
    {
      enum Overlap {vertex, aggregate, pairwise, none};

      Overlap overlap;
      bool onthefly;

      SeqOverlappingSchwarzSmootherArgs(Overlap overlap_=vertex,
                                        bool onthefly_=false)
        : overlap(overlap_), onthefly(onthefly_)
      {}
    };

    template<class M, class X, class TM, class TS, class TA>
    struct SmootherTraits<SeqOverlappingSchwarz<M,X,TM,TS,TA> >
    {
      typedef  SeqOverlappingSchwarzSmootherArgs<typename M::field_type> Arguments;
    };

    template<class M, class X, class TM, class TS, class TA>
    class ConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TS,TA> >
      : public DefaultConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TS,TA> >
    {
      typedef DefaultConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TS,TA> > Father;

    public:
      typedef typename MatrixGraph<M>::VertexDescriptor VertexDescriptor;
      typedef Dune::Amg::AggregatesMap<VertexDescriptor> AggregatesMap;
      typedef typename AggregatesMap::AggregateDescriptor AggregateDescriptor;
      typedef typename SeqOverlappingSchwarz<M,X,TM,TS,TA>::subdomain_vector Vector;
      typedef typename Vector::value_type Subdomain;

      virtual void setMatrix(const M& matrix, const AggregatesMap& amap)
      {
        Father::setMatrix(matrix);

        std::vector<bool> visited(amap.noVertices(), false);
        typedef IteratorPropertyMap<std::vector<bool>::iterator,IdentityMap> VisitedMapType;
        VisitedMapType visitedMap(visited.begin());

        MatrixGraph<const M> graph(matrix);

        typedef SeqOverlappingSchwarzSmootherArgs<typename M::field_type> SmootherArgs;

        switch(Father::getArgs().overlap) {
        case SmootherArgs::vertex :
        {
          VertexAdder visitor(subdomains, amap);
          createSubdomains(matrix, graph, amap, visitor,  visitedMap);
        }
        break;
        case SmootherArgs::pairwise :
        {
          createPairDomains(graph);
        }
        break;
        case SmootherArgs::aggregate :
        {
          AggregateAdder<VisitedMapType> visitor(subdomains, amap, graph, visitedMap);
          createSubdomains(matrix, graph, amap, visitor, visitedMap);
        }
        break;
        case SmootherArgs::none :
          NoneAdder visitor;
          createSubdomains(matrix, graph, amap, visitor, visitedMap);
          break;
        default :
          DUNE_THROW(NotImplemented, "This overlapping scheme is not supported!");
        }
      }
      void setMatrix(const M& matrix)
      {
        Father::setMatrix(matrix);

        /* Create aggregates map where each aggregate is just one vertex. */
        AggregatesMap amap(matrix.N());
        VertexDescriptor v=0;
        for(typename AggregatesMap::iterator iter=amap.begin();
            iter!=amap.end(); ++iter)
          *iter=v++;

        std::vector<bool> visited(amap.noVertices(), false);
        typedef IteratorPropertyMap<std::vector<bool>::iterator,IdentityMap> VisitedMapType;
        VisitedMapType visitedMap(visited.begin());

        MatrixGraph<const M> graph(matrix);

        typedef SeqOverlappingSchwarzSmootherArgs<typename M::field_type> SmootherArgs;

        switch(Father::getArgs().overlap) {
        case SmootherArgs::vertex :
        {
          VertexAdder visitor(subdomains, amap);
          createSubdomains(matrix, graph, amap, visitor,  visitedMap);
        }
        break;
        case SmootherArgs::aggregate :
        {
          DUNE_THROW(NotImplemented, "Aggregate overlap is not supported yet");
          /*
             AggregateAdder<VisitedMapType> visitor(subdomains, amap, graph, visitedMap);
             createSubdomains(matrix, graph, amap, visitor, visitedMap);
           */
        }
        break;
        case SmootherArgs::pairwise :
        {
          createPairDomains(graph);
        }
        break;
        case SmootherArgs::none :
          NoneAdder visitor;
          createSubdomains(matrix, graph, amap, visitor, visitedMap);

        }
      }

      const Vector& getSubDomains()
      {
        return subdomains;
      }

    private:
      struct VertexAdder
      {
        VertexAdder(Vector& subdomains_, const AggregatesMap& aggregates_)
          : subdomains(subdomains_), max(-1), subdomain(-1), aggregates(aggregates_)
        {}
        template<class T>
        void operator()(const T& edge)
        {
          if(aggregates[edge.target()]!=AggregatesMap::ISOLATED)
            subdomains[subdomain].insert(edge.target());
        }
        int setAggregate(const AggregateDescriptor& aggregate_)
        {
          subdomain=aggregate_;
          max = std::max(subdomain, aggregate_);
          return subdomain;
        }
        int noSubdomains() const
        {
          return max+1;
        }
      private:
        Vector& subdomains;
        AggregateDescriptor max;
        AggregateDescriptor subdomain;
        const AggregatesMap& aggregates;
      };
      struct NoneAdder
      {
        template<class T>
        void operator()(const T& edge)
        {}
        int setAggregate(const AggregateDescriptor& aggregate_)
        {
          return -1;
        }
        int noSubdomains() const
        {
          return -1;
        }
      };

      template<class VM>
      struct AggregateAdder
      {
        AggregateAdder(Vector& subdomains_, const AggregatesMap& aggregates_,
                       const MatrixGraph<const M>& graph_, VM& visitedMap_)
          : subdomains(subdomains_), subdomain(-1), aggregates(aggregates_),
            adder(subdomains_, aggregates_), graph(graph_), visitedMap(visitedMap_)
        {}
        template<class T>
        void operator()(const T& edge)
        {
          subdomains[subdomain].insert(edge.target());
          // If we (the neighbouring vertex of the aggregate)
          // are not isolated, add the aggregate we belong to
          // to the same subdomain using the OneOverlapAdder
          if(aggregates[edge.target()]!=AggregatesMap::ISOLATED) {
            assert(aggregates[edge.target()]!=aggregate);
            typename AggregatesMap::VertexList vlist;
            aggregates.template breadthFirstSearch<true,false>(edge.target(), aggregate,
                                                               graph, vlist, adder, adder,
                                                               visitedMap);
          }
        }

        int setAggregate(const AggregateDescriptor& aggregate_)
        {
          adder.setAggregate(aggregate_);
          aggregate=aggregate_;
          return ++subdomain;
        }
        int noSubdomains() const
        {
          return subdomain+1;
        }

      private:
        AggregateDescriptor aggregate;
        Vector& subdomains;
        int subdomain;
        const AggregatesMap& aggregates;
        VertexAdder adder;
        const MatrixGraph<const M>& graph;
        VM& visitedMap;
      };

      void createPairDomains(const MatrixGraph<const M>& graph)
      {
        typedef typename MatrixGraph<const M>::ConstVertexIterator VIter;
        typedef typename MatrixGraph<const M>::ConstEdgeIterator EIter;
        typedef typename M::size_type size_type;

        std::set<std::pair<size_type,size_type> > pairs;
        int total=0;
        for(VIter v=graph.begin(), ve=graph.end(); ve != v; ++v)
          for(EIter e = v.begin(), ee=v.end(); ee!=e; ++e)
          {
            ++total;
            if(e.source()<e.target())
              pairs.insert(std::make_pair(e.source(),e.target()));
            else
              pairs.insert(std::make_pair(e.target(),e.source()));
          }


        subdomains.resize(pairs.size());
        Dune::dinfo <<std::endl<< "Created "<<pairs.size()<<" ("<<total<<") pair domains"<<std::endl<<std::endl;
        typedef typename std::set<std::pair<size_type,size_type> >::const_iterator SIter;
        typename Vector::iterator subdomain=subdomains.begin();

        for(SIter s=pairs.begin(), se =pairs.end(); se!=s; ++s)
        {
          subdomain->insert(s->first);
          subdomain->insert(s->second);
          ++subdomain;
        }
        std::size_t minsize=10000;
        std::size_t maxsize=0;
        int sum=0;
        for(typename Vector::size_type i=0; i < subdomains.size(); ++i) {
          sum+=subdomains[i].size();
          minsize=std::min(minsize, subdomains[i].size());
          maxsize=std::max(maxsize, subdomains[i].size());
        }
        Dune::dinfo<<"Subdomain size: min="<<minsize<<" max="<<maxsize<<" avg="<<(sum/subdomains.size())
                   <<" no="<<subdomains.size()<<std::endl;
      }

      template<class Visitor>
      void createSubdomains(const M& matrix, const MatrixGraph<const M>& graph,
                            const AggregatesMap& amap, Visitor& overlapVisitor,
                            IteratorPropertyMap<std::vector<bool>::iterator,IdentityMap>& visitedMap )
      {
        // count  number ag aggregates. We assume that the
        // aggregates are numbered consecutively from 0 except
        // for the isolated ones. All isolated vertices form
        // one aggregate, here.
        int isolated=0;
        AggregateDescriptor maxAggregate=0;

        for(std::size_t i=0; i < amap.noVertices(); ++i)
          if(amap[i]==AggregatesMap::ISOLATED)
            isolated++;
          else
            maxAggregate = std::max(maxAggregate, amap[i]);

        subdomains.resize(maxAggregate+1+isolated);

        // reset the subdomains
        for(typename Vector::size_type i=0; i < subdomains.size(); ++i)
          subdomains[i].clear();

        // Create the subdomains from the aggregates mapping.
        // For each aggregate we mark all entries and the
        // neighbouring vertices as belonging to the same subdomain
        VertexAdder aggregateVisitor(subdomains, amap);

        for(VertexDescriptor i=0; i < amap.noVertices(); ++i)
          if(!get(visitedMap, i)) {
            AggregateDescriptor aggregate=amap[i];

            if(amap[i]==AggregatesMap::ISOLATED) {
              // isolated vertex gets its own aggregate
              subdomains.push_back(Subdomain());
              aggregate=subdomains.size()-1;
            }
            overlapVisitor.setAggregate(aggregate);
            aggregateVisitor.setAggregate(aggregate);
            subdomains[aggregate].insert(i);
            typename AggregatesMap::VertexList vlist;
            amap.template breadthFirstSearch<false,false>(i, aggregate, graph, vlist, aggregateVisitor,
                                                          overlapVisitor, visitedMap);
          }

        std::size_t minsize=10000;
        std::size_t maxsize=0;
        int sum=0;
        for(typename Vector::size_type i=0; i < subdomains.size(); ++i) {
          sum+=subdomains[i].size();
          minsize=std::min(minsize, subdomains[i].size());
          maxsize=std::max(maxsize, subdomains[i].size());
        }
        Dune::dinfo<<"Subdomain size: min="<<minsize<<" max="<<maxsize<<" avg="<<(sum/subdomains.size())
                   <<" no="<<subdomains.size()<<" isolated="<<isolated<<std::endl;



      }
      Vector subdomains;
    };


    template<class M, class X, class TM, class TS, class TA>
    struct ConstructionTraits<SeqOverlappingSchwarz<M,X,TM,TS,TA> >
    {
      typedef ConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TS,TA> > Arguments;

      static inline std::shared_ptr<SeqOverlappingSchwarz<M,X,TM,TS,TA>> construct(Arguments& args)
      {
        return std::make_shared<SeqOverlappingSchwarz<M,X,TM,TS,TA>>
          (args.getMatrix(),
           args.getSubDomains(),
           args.getArgs().relaxationFactor,
           args.getArgs().onthefly);
      }
    };


  } // namespace Amg
} // namespace Dune



#endif
