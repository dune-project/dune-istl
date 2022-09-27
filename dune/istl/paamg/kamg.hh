// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_KAMG_HH
#define DUNE_AMG_KAMG_HH

#include <dune/istl/preconditioners.hh>
#include "amg.hh"

namespace Dune
{
  namespace Amg
  {

    /**
     * @addtogroup ISTL_PAAMG
     * @{
     */
    /** @file
     * @author Markus Blatt
     * @brief Provides an algebraic multigrid using a Krylov cycle.
     *
     */

    /**
     * @brief Two grid operator for AMG with Krylov cycle.
     * @tparam AMG The type of the underlying agglomeration AMG.
     */
    template<class AMG>
    class KAmgTwoGrid
      : public Preconditioner<typename AMG::Domain,typename AMG::Range>
    {
      /** @brief The type of the domain. */
      typedef typename AMG::Domain Domain;
      /** @brief the type of the range. */
      typedef typename AMG::Range Range;
    public:

      //! Category of the preconditioner (see SolverCategory::Category)
      virtual SolverCategory::Category category() const
      {
        return amg_.category();
      };

      /**
       * @brief Constructor.
       * @param amg The underlying amg. It is used as the storage for the hierarchic
       * data structures.
       * @param coarseSolver The solver used for the coarse grid correction.
       */

      KAmgTwoGrid(AMG& amg, std::shared_ptr<InverseOperator<Domain,Range> > coarseSolver)
        : amg_(amg), coarseSolver_(coarseSolver)
      {}

      /**  \copydoc Preconditioner::pre(X&,Y&) */
      void pre([[maybe_unused]] typename AMG::Domain& x, [[maybe_unused]] typename AMG::Range& b)
      {}

      /**  \copydoc Preconditioner::post(X&) */
      void post([[maybe_unused]] typename AMG::Domain& x)
      {}

      /** \copydoc Preconditioner::apply(X&,const Y&) */
      void apply(typename AMG::Domain& v, const typename AMG::Range& d)
      {
        // Copy data
        *levelContext_->update=0;
        *levelContext_->rhs = d;
        *levelContext_->lhs = v;

        presmooth(*levelContext_, amg_.preSteps_);
        bool processFineLevel =
          amg_.moveToCoarseLevel(*levelContext_);

        if(processFineLevel) {
          typename AMG::Range b=*levelContext_->rhs;
          typename AMG::Domain x=*levelContext_->update;
          InverseOperatorResult res;
          coarseSolver_->apply(x, b, res);
          *levelContext_->update=x;
        }

        amg_.moveToFineLevel(*levelContext_, processFineLevel);

        postsmooth(*levelContext_, amg_.postSteps_);
        v=*levelContext_->update;
      }

      /**
       * @brief Get a pointer to the coarse grid solver.
       * @return The coarse grid solver.
       */
      InverseOperator<Domain,Range>* coarseSolver()
      {
        return coarseSolver_;
      }

      /**
       * @brief Set the level context pointer.
       * @param p The pointer to set it to.
       */
      void setLevelContext(std::shared_ptr<typename AMG::LevelContext> p)
      {
        levelContext_=p;
      }

      /** @brief Destructor. */
      ~KAmgTwoGrid()
      {}

    private:
      /** @brief Underlying AMG used as storage and engine. */
      AMG& amg_;
      /** @brief The coarse grid solver.*/
      std::shared_ptr<InverseOperator<Domain,Range> > coarseSolver_;
      /** @brief A shared pointer to the level context of AMG. */
      std::shared_ptr<typename AMG::LevelContext> levelContext_;
    };



    /**
     * @brief an algebraic multigrid method using a Krylov-cycle.
     *
     * The implementation is based on the paper
     * [[Notay and Vassilevski, 2007]](http://onlinelibrary.wiley.com/doi/10.1002/nla.542/abstract)
     *
     * @tparam M The type of the linear operator.
     * @tparam X The type of the range and domain.
     * @tparam PI The parallel information object. Use SequentialInformation (default)
     * for a sequential AMG, OwnerOverlapCopyCommunication for the parallel case.
     * @tparam K The type of the Krylov method to use for the cycle.
     * @tparam A The type of the allocator to use.
     */
    template<class M, class X, class S, class PI=SequentialInformation,
        class K=GeneralizedPCGSolver<X>, class A=std::allocator<X> >
    class KAMG : public Preconditioner<X,X>
    {
    public:
      /** @brief The type of the underlying AMG. */
      typedef AMG<M,X,S,PI,A> Amg;
      /** @brief The type of the Krylov solver for the cycle. */
      typedef K KrylovSolver;
      /** @brief The type of the hierarchy of operators. */
      typedef typename Amg::OperatorHierarchy OperatorHierarchy;
      /** @brief The type of the coarse solver. */
      typedef typename Amg::CoarseSolver CoarseSolver;
      /** @brief the type of the parallelinformation to use.*/
      typedef typename Amg::ParallelInformation ParallelInformation;
      /** @brief The type of the arguments for construction of the smoothers. */
      typedef typename Amg::SmootherArgs SmootherArgs;
      /** @brief the type of the lineatr operator. */
      typedef typename Amg::Operator Operator;
      /** @brief the type of the domain. */
      typedef typename Amg::Domain Domain;
      /** @brief The type of the range. */
      typedef typename Amg::Range Range;
      /** @brief The type of the hierarchy of parallel information. */
      typedef typename Amg::ParallelInformationHierarchy ParallelInformationHierarchy;
      /** @brief The type of the scalar product. */
      typedef typename Amg::ScalarProduct ScalarProduct;

      //! Category of the preconditioner (see SolverCategory::Category)
      virtual SolverCategory::Category category() const
      {
        return amg.category();
      };

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse
       * grid, must match the coarse matrix in the matrix hierarchy.
       * @param smootherArgs The  arguments needed for thesmoother to use
       * for pre and post smoothing.
       * @param parms The parameters for the AMG.
       * @param maxLevelKrylovSteps maximum of krylov iterations on a particular level (default=3)
       * @param minDefectReduction minimal defect reduction during the krylov iterations on a particular level (default=1e-1)
       */
      KAMG(OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
           const SmootherArgs& smootherArgs, const Parameters& parms,
           std::size_t maxLevelKrylovSteps=3, double minDefectReduction=1e-1);

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion, and providing the parameters.
       * @param smootherArgs The arguments for constructing the smoothers.
       * @param maxLevelKrylovSteps maximum of krylov iterations on a particular level (default=3)
       * @param minDefectReduction minimal defect reduction during the krylov iterations on a particular level (default=1e-1)
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      KAMG(const Operator& fineOperator, const C& criterion,
           const SmootherArgs& smootherArgs=SmootherArgs(),
           std::size_t maxLevelKrylovSteps=3, double minDefectReduction=1e-1,
           const ParallelInformation& pinfo=ParallelInformation());

      /**  \copydoc Preconditioner::pre(X&,Y&) */
      void pre(Domain& x, Range& b);
      /**  \copydoc Preconditioner::post(X&) */
      void post(Domain& x);
      /**  \copydoc Preconditioner::apply(X&,const Y&) */
      void apply(Domain& v, const Range& d);

      std::size_t maxlevels();

    private:
      /** @brief The underlying amg. */
      Amg amg;

      /** \brief The maximum number of Krylov steps allowed at each level. */
      std::size_t maxLevelKrylovSteps;

      /** \brief The defect reduction to achieve on each krylov level. */
      double levelDefectReduction;

      /** @brief pointers to the allocated scalar products. */
      std::vector<std::shared_ptr<typename Amg::ScalarProduct> > scalarproducts;

      /** @brief pointers to the allocated krylov solvers. */
      std::vector<std::shared_ptr<KAmgTwoGrid<Amg> > > ksolvers;
    };


    template<class M, class X, class S, class P, class K, class A>
    KAMG<M,X,S,P,K,A>::KAMG(OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
                            const SmootherArgs& smootherArgs, const Parameters& params,
                            std::size_t ksteps, double reduction)
      : amg(matrices, coarseSolver, smootherArgs, params),
        maxLevelKrylovSteps(ksteps), levelDefectReduction(reduction)
    {}


    template<class M, class X, class S, class P, class K, class A>
    template<class C>
    KAMG<M,X,S,P,K,A>::KAMG(const Operator& fineOperator, const C& criterion,
                            const SmootherArgs& smootherArgs,
                            std::size_t ksteps, double reduction,
                            const ParallelInformation& pinfo)
      : amg(fineOperator, criterion, smootherArgs, pinfo),
        maxLevelKrylovSteps(ksteps), levelDefectReduction(reduction)
    {}


    template<class M, class X, class S, class P, class K, class A>
    void KAMG<M,X,S,P,K,A>::pre(Domain& x, Range& b)
    {
      amg.pre(x,b);
      scalarproducts.reserve(amg.levels());
      ksolvers.reserve(amg.levels());

      typename OperatorHierarchy::ParallelMatrixHierarchy::Iterator
      matrix = amg.matrices_->matrices().coarsest();
      typename ParallelInformationHierarchy::Iterator
      pinfo = amg.matrices_->parallelInformation().coarsest();
      bool hasCoarsest=(amg.levels()==amg.maxlevels());

      if(hasCoarsest) {
        if(matrix==amg.matrices_->matrices().finest())
          return;
        --matrix;
        --pinfo;
        ksolvers.push_back(std::shared_ptr<KAmgTwoGrid<Amg> >(new KAmgTwoGrid<Amg>(amg, amg.solver_)));
      }else
        ksolvers.push_back(std::shared_ptr<KAmgTwoGrid<Amg> >(new KAmgTwoGrid<Amg>(amg, std::shared_ptr<InverseOperator<Domain,Range> >())));

      std::ostringstream s;

      if(matrix!=amg.matrices_->matrices().finest())
        while(true) {
          scalarproducts.push_back(createScalarProduct<X>(*pinfo,category()));
          std::shared_ptr<InverseOperator<Domain,Range> > ks =
            std::shared_ptr<InverseOperator<Domain,Range> >(new KrylovSolver(*matrix, *(scalarproducts.back()),
                                                                        *(ksolvers.back()), levelDefectReduction,
                                                                        maxLevelKrylovSteps, 0));
          ksolvers.push_back(std::shared_ptr<KAmgTwoGrid<Amg> >(new KAmgTwoGrid<Amg>(amg, ks)));
          --matrix;
          --pinfo;
          if(matrix==amg.matrices_->matrices().finest())
            break;
        }
    }


    template<class M, class X, class S, class P, class K, class A>
    void KAMG<M,X,S,P,K,A>::post(Domain& x)
    {
      amg.post(x);

    }

    template<class M, class X, class S, class P, class K, class A>
    void KAMG<M,X,S,P,K,A>::apply(Domain& v, const Range& d)
    {
      if(ksolvers.size()==0)
      {
        Range td=d;
        InverseOperatorResult res;
        amg.solver_->apply(v,td,res);
      }else
      {
        typedef typename Amg::LevelContext LevelContext;
        std::shared_ptr<LevelContext> levelContext(new LevelContext);
        amg.initIteratorsWithFineLevel(*levelContext);
        typedef typename std::vector<std::shared_ptr<KAmgTwoGrid<Amg> > >::iterator Iter;
        for(Iter solver=ksolvers.begin(); solver!=ksolvers.end(); ++solver)
          (*solver)->setLevelContext(levelContext);
        ksolvers.back()->apply(v,d);
      }
    }

    template<class M, class X, class S, class P, class K, class A>
    std::size_t KAMG<M,X,S,P,K,A>::maxlevels()
    {
      return amg.maxlevels();
    }

    /** @}*/
  } // Amg
} // Dune

#endif
