// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_AMG_AMG_HH
#define DUNE_AMG_AMG_HH

#include <memory>
#include <dune/common/exceptions.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/paamg/transfer.hh>
#include <dune/istl/paamg/hierarchy.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/superlu.hh>
#include <dune/common/typetraits.hh>

namespace Dune
{
  namespace Amg
  {
    /**
     * @defgroup ISTL_PAAMG Parallel Algebraic Multigrid
     * @ingroup ISTL_Prec
     * @brief A Parallel Algebraic Multigrid based on Agglomeration.
     */

    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */

    /** @file
     * @author Markus Blatt
     * @brief The AMG preconditioner.
     */

    /**
     * @brief Parallel algebraic multigrid based on agglomeration.
     *
     * \tparam M The matrix type
     * \tparam X The vector type
     * \tparam A An allocator for X
     */
    template<class M, class X, class S, class PI=SequentialInformation,
        class A=std::allocator<X> >
    class AMG : public Preconditioner<X,X>
    {
    public:
      /** @brief The matrix operator type. */
      typedef M Operator;
      /**
       * @brief The type of the parallel information.
       * Either OwnerOverlapCommunication or another type
       * describing the parallel data distribution and
       * providing communication methods.
       */
      typedef PI ParallelInformation;
      /** @brief The operator hierarchy type. */
      typedef MatrixHierarchy<M, ParallelInformation, A> OperatorHierarchy;
      /** @brief The parallal data distribution hierarchy type. */
      typedef typename OperatorHierarchy::ParallelInformationHierarchy ParallelInformationHierarchy;

      /** @brief The domain type. */
      typedef X Domain;
      /** @brief The range type. */
      typedef X Range;
      /** @brief the type of the coarse solver. */
      typedef InverseOperator<X,X> CoarseSolver;
      /**
       * @brief The type of the smoother.
       *
       * One of the preconditioners implementing the Preconditioner interface.
       * Note that the smoother has to fit the ParallelInformation.*/
      typedef S Smoother;

      /** @brief The argument type for the construction of the smoother. */
      typedef typename SmootherTraits<Smoother>::Arguments SmootherArgs;

      enum {
        /** @brief The solver category. */
        category = S::category
      };

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse
       * grid, must natch the soarse matrix in the matrix hierachy.
       * @param smootherArgs The  arguments needed for thesmoother to use
       * for pre and post smoothing
       * @param gamma The number of subcycles. 1 for V-cycle, 2 for W-cycle.
       * @param preSmoothingSteps The number of smoothing steps for premoothing.
       * @param postSmoothingSteps The number of smoothing steps for postmoothing.
       */
      AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
          const SmootherArgs& smootherArgs, std::size_t gamma,
          std::size_t preSmoothingSteps,
          std::size_t postSmoothingSteps, bool additive=false);

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion.
       * @param smootherArgs The arguments for constructing the smoothers.
       * @param gamma 1 for V-cycle, 2 for W-cycle
       * @param preSmoothingSteps The number of smoothing steps for premoothing.
       * @param postSmoothingSteps The number of smoothing steps for postmoothing.
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      AMG(const Operator& fineOperator, const C& criterion,
          const SmootherArgs& smootherArgs, std::size_t gamma=1,
          std::size_t preSmoothingSteps=2, std::size_t postSmoothingSteps=2,
          bool additive=false, const ParallelInformation& pinfo=ParallelInformation());

      ~AMG();

      /** \copydoc Preconditioner::pre */
      void pre(Domain& x, Range& b);

      /** \copydoc Preconditioner::apply */
      void apply(Domain& v, const Range& d);

      /** \copydoc Preconditioner::post */
      void post(Domain& x);

      /**
       * @brief Get the aggregate number of each unknown on the coarsest level.
       * @param cont The random access container to store the numbers in.
       */
      template<class A1>
      void getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont);

      std::size_t levels();

      std::size_t maxlevels();
    private:
      /** @brief Multigrid cycle on a level. */
      void mgc(typename Hierarchy<Smoother,A>::Iterator& smoother,
               typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator& matrix,
               typename ParallelInformationHierarchy::Iterator& pinfo,
               typename OperatorHierarchy::RedistributeInfoList::const_iterator& redist,
               typename OperatorHierarchy::AggregatesMapList::const_iterator& aggregates,
               typename Hierarchy<Domain,A>::Iterator& lhs,
               typename Hierarchy<Domain,A>::Iterator& update,
               typename Hierarchy<Range,A>::Iterator& rhs);

      void additiveMgc();

      //      void setupIndices(typename Matrix::ParallelIndexSet& indices, const Matrix& matrix);

      /**  @brief The matrix we solve. */
      const OperatorHierarchy* matrices_;
      /** @brief The arguments to construct the smoother */
      SmootherArgs smootherArgs_;
      /** @brief The hierarchy of the smoothers. */
      Hierarchy<Smoother,A> smoothers_;
      /** @brief The solver of the coarsest level. */
      CoarseSolver* solver_;
      /** @brief The right hand side of our problem. */
      Hierarchy<Range,A>* rhs_;
      /** @brief The left approximate solution of our problem. */
      Hierarchy<Domain,A>* lhs_;
      /** @brief The total update for the outer solver. */
      Hierarchy<Domain,A>* update_;
      /** @brief The type of the chooser of the scalar product. */
      typedef Dune::ScalarProductChooser<X,PI,M::category> ScalarProductChooser;
      /** @brief The type of the scalar product for the coarse solver. */
      typedef typename ScalarProductChooser::ScalarProduct ScalarProduct;
      /** @brief Scalar product on the coarse level. */
      ScalarProduct* scalarProduct_;
      /** @brief Gamma, 1 for V-cycle and 2 for W-cycle. */
      std::size_t gamma_;
      /** @brief The number of pre and postsmoothing steps. */
      std::size_t preSteps_;
      /** @brief The number of postsmoothing steps. */
      std::size_t postSteps_;
      std::size_t level;
      bool buildHierarchy_;
      bool additive;
      Smoother *coarseSmoother_;
    };

    template<class M, class X, class S, class P, class A>
    AMG<M,X,S,P,A>::AMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
                        const SmootherArgs& smootherArgs,
                        std::size_t gamma, std::size_t preSmoothingSteps,
                        std::size_t postSmoothingSteps, bool additive_)
      : matrices_(&matrices), smootherArgs_(smootherArgs),
        smoothers_(), solver_(&coarseSolver), scalarProduct_(0),
        gamma_(gamma), preSteps_(preSmoothingSteps), postSteps_(postSmoothingSteps), buildHierarchy_(false),
        additive(additive_), coarseSmoother_()
    {
      assert(matrices_->isBuilt());

      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);
    }

    template<class M, class X, class S, class P, class A>
    template<class C>
    AMG<M,X,S,P,A>::AMG(const Operator& matrix,
                        const C& criterion,
                        const SmootherArgs& smootherArgs,
                        std::size_t gamma, std::size_t preSmoothingSteps,
                        std::size_t postSmoothingSteps,
                        bool additive_,
                        const P& pinfo)
      : smootherArgs_(smootherArgs),
        smoothers_(), solver_(), scalarProduct_(0), gamma_(gamma),
        preSteps_(preSmoothingSteps), postSteps_(postSmoothingSteps), buildHierarchy_(true),
        additive(additive_), coarseSmoother_()
    {
      dune_static_assert(static_cast<int>(M::category)==static_cast<int>(S::category),
                         "Matrix and Solver must match in terms of category!");
      dune_static_assert(static_cast<int>(P::category)==static_cast<int>(S::category),
                         "Matrix and Solver must match in terms of category!");
      Timer watch;
      OperatorHierarchy* matrices = new OperatorHierarchy(const_cast<Operator&>(matrix), pinfo);

      matrices->template build<typename P::CopyFlags>(criterion);

      matrices_ = matrices;
      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(smoothers_, smootherArgs_);

      if(criterion.debugLevel()>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
        std::cout<<"Building Hierarchy of "<<matrices_->maxlevels()<<" levels took "<<watch.elapsed()<<" seconds."<<std::endl;
    }

    template<class M, class X, class S, class P, class A>
    AMG<M,X,S,P,A>::~AMG()
    {
      if(buildHierarchy_) {
        delete matrices_;
      }
      if(scalarProduct_)
        delete scalarProduct_;
    }

    template<class M, class X, class S, class P, class A>
    void AMG<M,X,S,P,A>::pre(Domain& x, Range& b)
    {
      Range* copy = new Range(b);
      rhs_ = new Hierarchy<Range,A>(*copy);
      Domain* dcopy = new Domain(x);
      lhs_ = new Hierarchy<Domain,A>(*dcopy);
      dcopy = new Domain(x);
      update_ = new Hierarchy<Domain,A>(*dcopy);
      matrices_->coarsenVector(*rhs_);
      matrices_->coarsenVector(*lhs_);
      matrices_->coarsenVector(*update_);

      // Preprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Range,A>::Iterator RIterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_.coarsest();
      Iterator smoother = smoothers_.finest();
      RIterator rhs = rhs_->finest();
      DIterator lhs = lhs_->finest();
      if(smoothers_.levels()>0) {

        assert(lhs_->levels()==rhs_->levels());
        assert(smoothers_.levels()==lhs_->levels() || matrices_->levels()==matrices_->maxlevels());
        assert(smoothers_.levels()+1==lhs_->levels() || matrices_->levels()<matrices_->maxlevels());

        if(rhs!=rhs_->coarsest() || matrices_->levels()<matrices_->maxlevels())
          smoother->pre(*lhs,*rhs);

        if(smoother!=coarsest)
          for(++smoother, ++lhs, ++rhs; smoother != coarsest; ++smoother, ++lhs, ++rhs)
            smoother->pre(*lhs,*rhs);
        smoother->pre(*lhs,*rhs);
      }


      // The preconditioner might change x and b. So we have to
      // copy the changes to the original vectors.
      x = *lhs_->finest();
      b = *rhs_->finest();

      if(buildHierarchy_ && matrices_->levels()==matrices_->maxlevels()) {
        // We have the carsest level. Create the coarse Solver
        SmootherArgs sargs(smootherArgs_);
        sargs.iterations = 1;

        typename ConstructionTraits<Smoother>::Arguments cargs;
        cargs.setArgs(sargs);
        if(matrices_->redistributeInformation().back().isSetup()) {
          // Solve on the redistributed partitioning
          cargs.setMatrix(matrices_->matrices().coarsest().getRedistributed().getmat());
          cargs.setComm(matrices_->parallelInformation().coarsest().getRedistributed());

          coarseSmoother_ = ConstructionTraits<Smoother>::construct(cargs);
          scalarProduct_ = ScalarProductChooser::construct(matrices_->parallelInformation().coarsest().getRedistributed());
        }else{
          cargs.setMatrix(matrices_->matrices().coarsest()->getmat());
          cargs.setComm(*matrices_->parallelInformation().coarsest());

          coarseSmoother_ = ConstructionTraits<Smoother>::construct(cargs);
          scalarProduct_ = ScalarProductChooser::construct(*matrices_->parallelInformation().coarsest());
        }
#ifdef HAVE_SUPERLU
        // Use superlu if we are purely sequential or with only one processor on the coarsest level.
        if(is_same<ParallelInformation,SequentialInformation>::value // sequential mode
           || matrices_->parallelInformation().coarsest()->communicator().size()==1 //parallel mode and only one processor
           || (matrices_->parallelInformation().coarsest().isRedistributed()
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()==1
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()>0)) { // redistribute and 1 proc
          std::cout<<"Using superlu"<<std::endl;
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
              // We are still participating on this level
              solver_  = new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest().getRedistributed().getmat());
            else
              solver_ = 0;
          }else
            solver_  = new SuperLU<typename M::matrix_type>(matrices_->matrices().coarsest()->getmat());
        }else
#endif
        {
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
              // We are still participating on this level
              solver_ = new BiCGSTABSolver<X>(const_cast<M&>(matrices_->matrices().coarsest().getRedistributed()),
                                              *scalarProduct_,
                                              *coarseSmoother_, 1E-2, 10000, 0);
            else
              solver_ = 0;
          }else
            solver_ = new BiCGSTABSolver<X>(const_cast<M&>(*matrices_->matrices().coarsest()),
                                            *scalarProduct_,
                                            *coarseSmoother_, 1E-2, 10000, 0);
        }
      }
    }
    template<class M, class X, class S, class P, class A>
    std::size_t AMG<M,X,S,P,A>::levels()
    {
      return matrices_->levels();
    }
    template<class M, class X, class S, class P, class A>
    std::size_t AMG<M,X,S,P,A>::maxlevels()
    {
      return matrices_->maxlevels();
    }

    /** \copydoc Preconditioner::apply */
    template<class M, class X, class S, class P, class A>
    void AMG<M,X,S,P,A>::apply(Domain& v, const Range& d)
    {
      if(additive) {
        *(rhs_->finest())=d;
        additiveMgc();
        v=*lhs_->finest();
      }else{
        typename Hierarchy<Smoother,A>::Iterator smoother = smoothers_.finest();
        typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator matrix = matrices_->matrices().finest();
        typename ParallelInformationHierarchy::Iterator pinfo = matrices_->parallelInformation().finest();
        typename OperatorHierarchy::RedistributeInfoList::const_iterator redist =
          matrices_->redistributeInformation().begin();
        typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates = matrices_->aggregatesMaps().begin();
        typename Hierarchy<Domain,A>::Iterator lhs = lhs_->finest();
        typename Hierarchy<Domain,A>::Iterator update = update_->finest();
        typename Hierarchy<Range,A>::Iterator rhs = rhs_->finest();

        *lhs = v;
        *rhs = d;
        *update=0;
        level=0;

        mgc(smoother, matrix, pinfo, redist, aggregates, lhs, update, rhs);

        if(postSteps_==0||matrices_->maxlevels()==1)
          pinfo->copyOwnerToAll(*update, *update);

        v=*update;
      }

    }

    template<class M, class X, class S, class P, class A>
    void AMG<M,X,S,P,A>::mgc(typename Hierarchy<Smoother,A>::Iterator& smoother,
                             typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator& matrix,
                             typename ParallelInformationHierarchy::Iterator& pinfo,
                             typename OperatorHierarchy::RedistributeInfoList::const_iterator& redist,
                             typename OperatorHierarchy::AggregatesMapList::const_iterator& aggregates,
                             typename Hierarchy<Domain,A>::Iterator& lhs,
                             typename Hierarchy<Domain,A>::Iterator& update,
                             typename Hierarchy<Range,A>::Iterator& rhs){
      if(matrix == matrices_->matrices().coarsest() && levels()==maxlevels()) {
        // Solve directly
        InverseOperatorResult res;
        res.converged=true; // If we do not compute this flag will not get updated
        if(redist->isSetup()) {
          redist->redistribute(*rhs, rhs.getRedistributed());
          if(rhs.getRedistributed().size()>0) {
            // We are still participating in the computation
            pinfo.getRedistributed().copyOwnerToAll(rhs.getRedistributed(), rhs.getRedistributed());
            if(maxlevels()==1)
              // prepare for iterativr solver
              update.getRedistributed()=0.1;
            solver_->apply(update.getRedistributed(), rhs.getRedistributed(), res);
          }
          redist->redistributeBackward(*update, update.getRedistributed());
          pinfo->copyOwnerToAll(*update, *update);
        }else{
          pinfo->copyOwnerToAll(*rhs, *rhs);
          if(maxlevels()==1)
            // prepare for iterativr solver
            *update=0.1;
          solver_->apply(*update, *rhs, res);
        }

        if(!res.converged)
          DUNE_THROW(MathError, "Coarse solver did not converge");
      }else{
        // presmoothing
        *lhs=0;
        for(std::size_t i=0; i < preSteps_; ++i)
          SmootherApplier<S>::preSmooth(*smoother, *lhs, *rhs);

        // Accumulate update
        *update += *lhs;
        // update defect
        matrix->applyscaleadd(-1,static_cast<const Domain&>(*lhs), *rhs);

#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION

        bool processNextLevel=true;

        if(redist->isSetup()) {
          redist->redistribute(static_cast<const Range&>(*rhs), rhs.getRedistributed());
          processNextLevel =rhs.getRedistributed().size()>0;
          if(processNextLevel) {
            //restrict defect to coarse level right hand side.
            typename Hierarchy<Range,A>::Iterator fineRhs = rhs++;
            ++pinfo;
            Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
            ::restrict (*(*aggregates), *rhs, static_cast<const Range&>(fineRhs.getRedistributed()), *pinfo);
          }
        }else{
          //restrict defect to coarse level right hand side.
          typename Hierarchy<Range,A>::Iterator fineRhs = rhs++;
          ++pinfo;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::restrict (*(*aggregates), *rhs, static_cast<const Range&>(*fineRhs), *pinfo);
        }

        if(processNextLevel) {
          // prepare coarse system
          ++lhs;
          ++update;
          ++matrix;
          ++level;
          ++redist;

          if(matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
            // next level is not the globally coarsest one
            ++smoother;
            ++aggregates;
          }

          // prepare the update on the next level
          *update=0;

          // next level
          for(std::size_t i=0; i<gamma_; i++)
            mgc(smoother, matrix, pinfo, redist, aggregates, lhs, update, rhs);

          if(matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
            // previous level is not the globally coarsest one
            --smoother;
            --aggregates;
          }
          --redist;
          --level;
          //prolongate and add the correction (update is in coarse left hand side)
          --matrix;

          //typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--;
          --lhs;
          --pinfo;
        }

        if(redist->isSetup()) {
          // Need to redistribute during prolongate
          lhs.getRedistributed()=0;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::prolongate(*(*aggregates), *update, *lhs, lhs.getRedistributed(), matrices_->getProlongationDampingFactor(),
                       *pinfo, *redist);
        }else{
          *lhs=0;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::prolongate(*(*aggregates), *update, *lhs,
                       matrices_->getProlongationDampingFactor(), *pinfo);
        }


        if(processNextLevel) {
          --update;
          --rhs;
        }

        *update += *lhs;


#endif

        // update defect
        matrix->applyscaleadd(-1,static_cast<const Domain&>(*lhs), *rhs);

        // postsmoothing
        *lhs=0;
        pinfo->project(*rhs);

        for(std::size_t i=0; i < postSteps_; ++i)
          SmootherApplier<S>::postSmooth(*smoother, *lhs, *rhs);

        *update += *lhs;
      }
    }

    template<class M, class X, class S, class P, class A>
    void AMG<M,X,S,P,A>::additiveMgc(){

      // restrict residual to all levels
      typename ParallelInformationHierarchy::Iterator pinfo=matrices_->parallelInformation().finest();
      typename Hierarchy<Range,A>::Iterator rhs=rhs_->finest();
      typename Hierarchy<Domain,A>::Iterator lhs = lhs_->finest();
      typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates=matrices_->aggregatesMaps().begin();

      for(typename Hierarchy<Range,A>::Iterator fineRhs=rhs++; fineRhs != rhs_->coarsest(); fineRhs=rhs++, ++aggregates) {
        ++pinfo;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::restrict (*(*aggregates), *rhs, static_cast<const Range&>(*fineRhs), *pinfo);
      }

      // pinfo is invalid, set to coarsest level
      //pinfo = matrices_->parallelInformation().coarsest
      // calculate correction for all levels
      lhs = lhs_->finest();
      typename Hierarchy<Smoother,A>::Iterator smoother = smoothers_.finest();

      for(rhs=rhs_->finest(); rhs != rhs_->coarsest(); ++lhs, ++rhs, ++smoother) {
        // presmoothing
        *lhs=0;
        smoother->apply(*lhs, *rhs);
      }

      // Coarse level solve
#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION
      InverseOperatorResult res;
      pinfo->copyOwnerToAll(*rhs, *rhs);
      solver_->apply(*lhs, *rhs, res);

      if(!res.converged)
        DUNE_THROW(MathError, "Coarse solver did not converge");
#else
      *lhs=0;
#endif
      // Prologate and add up corrections from all levels
      --pinfo;
      --aggregates;

      for(typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--; coarseLhs != lhs_->finest(); coarseLhs = lhs--, --aggregates, --pinfo) {
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::prolongate(*(*aggregates), *coarseLhs, *lhs, 1, *pinfo);
      }
    }


    /** \copydoc Preconditioner::post */
    template<class M, class X, class S, class P, class A>
    void AMG<M,X,S,P,A>::post(Domain& x)
    {
      if(buildHierarchy_) {
        if(solver_)
          delete solver_;
        if(coarseSmoother_)
          ConstructionTraits<Smoother>::deconstruct(coarseSmoother_);
      }

      // Postprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Range,A>::Iterator RIterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_.coarsest();
      Iterator smoother = smoothers_.finest();
      DIterator lhs = lhs_->finest();
      if(smoothers_.levels()>0) {
        if(smoother != coarsest  || matrices_->levels()<matrices_->maxlevels())
          smoother->post(*lhs);
        if(smoother!=coarsest)
          for(++smoother, ++lhs; smoother != coarsest; ++smoother, ++lhs)
            smoother->post(*lhs);
        smoother->post(*lhs);
      }

      delete &(*lhs_->finest());
      delete lhs_;
      delete &(*update_->finest());
      delete update_;
      delete &(*rhs_->finest());
      delete rhs_;
    }

    template<class M, class X, class S, class P, class A>
    template<class A1>
    void AMG<M,X,S,P,A>::getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont)
    {
      matrices_->getCoarsestAggregatesOnFinest(cont);
    }

  } // end namespace Amg
} // end namespace Dune

#endif
