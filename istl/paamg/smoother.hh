// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGSMOOTHER_HH
#define DUNE_AMGSMOOTHER_HH

#include <dune/istl/paamg/construction.hh>
#include <dune/istl/paamg/aggregates.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/schwarz.hh>
#include <dune/common/propertymap.hh>

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
      typedef T RelaxationFactor;

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

    template<class X, class Y, class C, class T>
    struct SmootherTraits<BlockPreconditioner<X,Y,C,T> >
    {
      typedef DefaultSmootherArgs<typename T::matrix_type::field_type> Arguments;

    };


    /**
     * @brief Construction Arguments for the default smoothers
     */
    template<class T>
    class DefaultConstructionArgs
    {
      typedef typename T::matrix_type Matrix;

      typedef typename SmootherTraits<T>::Arguments SmootherArgs;

      typedef AggregatesMap<typename MatrixGraph<Matrix>::VertexDescriptor> AggregatesMap;

    public:
      virtual ~DefaultConstructionArgs()
      {}

      void setMatrix(const Matrix& matrix)
      {
        matrix_=&matrix;
      }
      virtual void setMatrix(const Matrix& matrix, const AggregatesMap& amap)
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
      void setComm(T1& comm)
      {}

      const SmootherArgs getArgs() const
      {
        return *args_;
      }

    protected:
      const Matrix* matrix_;
    private:
      const SmootherArgs* args_;
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


    template<class T>
    class ConstructionTraits;

    /**
     * @brief Policy for the construction of the SeqSSOR smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqSSOR<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqSSOR<M,X,Y,l> > Arguments;

      static inline SeqSSOR<M,X,Y,l>* construct(Arguments& args)
      {
        return new SeqSSOR<M,X,Y,l>(args.getMatrix(), args.getArgs().iterations,
                                    args.getArgs().relaxationFactor);
      }

      static inline void deconstruct(SeqSSOR<M,X,Y,l>* ssor)
      {
        delete ssor;
      }

    };


    /**
     * @brief Policy for the construction of the SeqSOR smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqSOR<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqSOR<M,X,Y,l> > Arguments;

      static inline SeqSOR<M,X,Y,l>* construct(Arguments& args)
      {
        return new SeqSOR<M,X,Y,l>(args.getMatrix(), args.getArgs().iterations,
                                   args.getArgs().relaxationFactor);
      }

      static inline void deconstruct(SeqSOR<M,X,Y,l>* sor)
      {
        delete sor;
      }

    };
    /**
     * @brief Policy for the construction of the SeqJac smoother
     */
    template<class M, class X, class Y, int l>
    struct ConstructionTraits<SeqJac<M,X,Y,l> >
    {
      typedef DefaultConstructionArgs<SeqJac<M,X,Y,l> > Arguments;

      static inline SeqJac<M,X,Y,l>* construct(Arguments& args)
      {
        return new SeqJac<M,X,Y,l>(args.getMatrix(), args.getArgs().iterations,
                                   args.getArgs().relaxationFactor);
      }

      static void deconstruct(SeqJac<M,X,Y,l>* jac)
      {
        delete jac;
      }

    };


    /**
     * @brief Policy for the construction of the SeqILUn smoother
     */
    template<class M, class X, class Y>
    struct ConstructionTraits<SeqILU0<M,X,Y> >
    {
      typedef DefaultConstructionArgs<SeqILU0<M,X,Y> > Arguments;

      static inline SeqILU0<M,X,Y>* construct(Arguments& args)
      {
        return new SeqILU0<M,X,Y>(args.getMatrix(),
                                  args.getArgs().relaxationFactor);
      }

      static void deconstruct(SeqILU0<M,X,Y>* ilu)
      {
        delete ilu;
      }

    };

    template<class M, class X, class Y>
    class ConstructionArgs<SeqILUn<M,X,Y> >
      : public DefaultConstructionArgs<SeqILUn<M,X,Y> >
    {
    public:
      ConstructionArgs(int n=1)
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
     * @brief Policy for the construction of the SeqJac smoother
     */
    template<class M, class X, class Y>
    struct ConstructionTraits<SeqILUn<M,X,Y> >
    {
      typedef ConstructionArgs<SeqILUn<M,X,Y> > Arguments;

      static inline SeqILUn<M,X,Y>* construct(Arguments& args)
      {
        return new SeqILUn<M,X,Y>(args.getMatrix(), args.getN(),
                                  args.getArgs().relaxationFactor);
      }

      static void deconstruct(SeqILUn<M,X,Y>* ilu)
      {
        delete ilu;
      }

    };


    /**
     * @brief Policy for the construction of the ParSSOR smoother
     */
    template<class M, class X, class Y, class C>
    struct ConstructionTraits<ParSSOR<M,X,Y,C> >
    {
      typedef DefaultParallelConstructionArgs<M,C> Arguments;

      static inline ParSSOR<M,X,Y,C>* construct(Arguments& args)
      {
        return new ParSSOR<M,X,Y,C>(args.getMatrix(), args.getArgs().iterations,
                                    args.getArgs().relaxationFactor,
                                    args.getComm());
      }
      static inline void deconstruct(ParSSOR<M,X,Y,C>* ssor)
      {
        delete ssor;
      }
    };

    template<class X, class Y, class C, class T>
    struct ConstructionTraits<BlockPreconditioner<X,Y,C,T> >
    {
      typedef DefaultParallelConstructionArgs<T,C> Arguments;
      typedef ConstructionTraits<T> SeqConstructionTraits;
      static inline BlockPreconditioner<X,Y,C,T>* construct(Arguments& args)
      {
        return new BlockPreconditioner<X,Y,C,T>(*SeqConstructionTraits::construct(args),
                                                args.getComm());
      }

      static inline void deconstruct(BlockPreconditioner<X,Y,C,T>* bp)
      {
        SeqConstructionTraits::deconstruct(static_cast<T*>(&bp->preconditioner));
        delete bp;
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
#ifdef HAVE_SUPERLU

    template<class M, class X, class TA>
    struct SmootherApplier<SeqOverlappingSchwarz<M,X,MultiplicativeSchwarzMode,TA> >
    {
      typedef SeqOverlappingSchwarz<M,X,MultiplicativeSchwarzMode,TA> Smoother;
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
      enum Overlap {vertex, aggregate, none};

      Overlap overlap;
      SeqOverlappingSchwarzSmootherArgs()
        : overlap(none)
      {}
    };

    template<class M, class X, class TM, class TA>
    struct SmootherTraits<SeqOverlappingSchwarz<M,X,TM,TA> >
    {
      typedef  SeqOverlappingSchwarzSmootherArgs<typename M::field_type> Arguments;
    };

    template<class M, class X, class TM, class TA>
    class ConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TA> >
      : public DefaultConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TA> >
    {
      typedef DefaultConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TA> > Father;

    public:
      typedef AggregatesMap<typename MatrixGraph<M>::VertexDescriptor> AggregatesMap;
      typedef typename AggregatesMap::AggregateDescriptor AggregateDescriptor;
      typedef typename AggregatesMap::VertexDescriptor VertexDescriptor;
      typedef typename SeqOverlappingSchwarz<M,X,TM,TA>::subdomain_vector Vector;

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
          VertexAdder visitor(subdomains);
          createSubdomains(matrix, graph, amap, visitor,  visitedMap);
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
        }
      }
      void setMatrix(const M& matrix)
      {
        Father::setMatrix(matrix);
      }

      const Vector& getSubDomains()
      {
        return subdomains;
      }

    private:
      struct VertexAdder
      {
        VertexAdder(Vector& subdomains_)
          : subdomains(subdomains_), subdomain(-1)
        {}
        template<class T>
        void operator()(const T& edge)
        {
          subdomains[subdomain].insert(edge.target());
        }
        int setAggregate(const AggregateDescriptor& aggregate_)
        {
          return ++subdomain;
        }
        int noSubdomains() const
        {
          return subdomain+1;
        }
      private:
        Vector& subdomains;
        int subdomain;
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
            adder(subdomains_), graph(graph_), visitedMap(visitedMap_)
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

      template<class Visitor>
      void createSubdomains(const M& matrix, const MatrixGraph<const M>& graph,
                            const AggregatesMap& amap, Visitor& overlapVisitor,
                            IteratorPropertyMap<std::vector<bool>::iterator,IdentityMap>& visitedMap )
      {
        // count  number ag aggregates. We asume that the
        // aggregates are numbered consecutively from 0 exept
        // for the isolated ones. All isolated vertices form
        // one aggregate, here.
        bool isolated=false;
        AggregateDescriptor maxAggregate=0;

        for(int i=0; i < amap.noVertices(); ++i)
          if(amap[i]==AggregatesMap::ISOLATED)
            isolated=true;
          else
            maxAggregate = std::max(maxAggregate, amap[i]);

        if(isolated)
          maxAggregate += 1;

        subdomains.resize(maxAggregate+1);

        // reset the subdomains
        for(int i=0; i < subdomains.size(); ++i)
          subdomains[i].clear();

        // Create the subdomains from the aggregates mapping.
        // For each aggregate we mark all entries and the
        // neighbouring vertices as belonging to the same subdomain
        VertexAdder aggregateVisitor(subdomains);

        for(VertexDescriptor i=0; i < amap.noVertices(); ++i)
          if(amap[i]!=AggregatesMap::ISOLATED && !get(visitedMap, i)) {
            AggregateDescriptor aggregate = (amap[i]==AggregatesMap::ISOLATED) ? maxAggregate : amap[i];
            overlapVisitor.setAggregate(aggregate);
            int domain=aggregateVisitor.setAggregate(aggregate);
            subdomains[domain].insert(i);
            typename AggregatesMap::VertexList vlist;
            amap.template breadthFirstSearch<false,false>(i, aggregate, graph, vlist, aggregateVisitor,
                                                          overlapVisitor, visitedMap);
          }
        subdomains.resize(aggregateVisitor.noSubdomains());

        std::size_t minsize=10000;
        std::size_t maxsize=0;
        int sum=0;
        for(int i=0; i < subdomains.size(); ++i) {
          sum+=subdomains[i].size();
          minsize=std::min(minsize, subdomains[i].size());
          maxsize=std::max(maxsize, subdomains[i].size());
        }
        std::cout<<"Subdomain size: min="<<minsize<<" max="<<maxsize<<" avg="<<(sum/subdomains.size())
                 <<" no="<<subdomains.size()<<std::endl;



      }
      Vector subdomains;
    };


    template<class M, class X, class TM, class TA>
    struct ConstructionTraits<SeqOverlappingSchwarz<M,X,TM,TA> >
    {
      typedef ConstructionArgs<SeqOverlappingSchwarz<M,X,TM,TA> > Arguments;

      static inline SeqOverlappingSchwarz<M,X,TM,TA>* construct(Arguments& args)
      {
        return new SeqOverlappingSchwarz<M,X,TM,TA>(args.getMatrix(),
                                                    args.getSubDomains(),
                                                    args.getArgs().relaxationFactor);
      }

      static void deconstruct(SeqOverlappingSchwarz<M,X,TM,TA>* schwarz)
      {
        delete schwarz;
      }
    };
#endif
  } // namespace Amg
} // namespace Dune



#endif
