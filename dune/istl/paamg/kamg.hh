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

      enum {
        /** @brief The solver category. */
        category = AMG::category
      };

      /**
       * @brief Constructor.
       * @param amg The underlying amg. It is used as the storage for the hierarchic
       * data structures.
       * @param coarseSolver The solver used for the coarse grid correction.
       */

      KAmgTwoGrid(AMG& amg, InverseOperator<Domain,Range>* coarseSolver, bool
                  isCoarsest)
        : amg_(amg), coarseSolver_(coarseSolver), isCoarsest_(isCoarsest)
      {}

      /**  \copydoc Preconditioner::pre(X&,Y&) */
      void pre(typename AMG::Domain& x, typename AMG::Range& b)
      {}

      /**  \copydoc Preconditioner::post(X&) */
      void post(typename AMG::Domain& x)
      {}

      /** \copydoc Preconditioner::apply(X&,const Y&) */
      void apply(typename AMG::Domain& v, const typename AMG::Range& d)
      {
        Range rhs(d);
        *amg_.lhs = v;
        // Copy data

        bool processFineLevel =
          amg_.moveToCoarseLevel(rhs);

        if(processFineLevel) {
          typename AMG::Domain x(*amg_.update=0); // update is zero
          if(!isCoarsest_)
            amg_.presmooth();
          typename AMG::Range b(*amg_.rhs); // b might get overridden
          InverseOperatorResult res;
          coarseSolver_->apply(x, b, res);
          *amg_.update+=x; // coarse level correction has to be in lhs

          if(!isCoarsest_) {
            *amg_.lhs=x;
            amg_.postsmooth();
          }
        }

        amg_.moveToFineLevel(v, processFineLevel);
      }

      /**
       * @brief Get a pointer to the coarse grid solver.
       * @return The coarse grid solver.
       */
      InverseOperator<Domain,Range>* coarseSolver()
      {
        return coarseSolver_;
      }

      /** @brief Destructor. */
      ~KAmgTwoGrid()
      {}

    private:
      /** @brief Underlying AMG used as storage and engine. */
      AMG& amg_;
      /** @brief The coarse grid solver.*/
      InverseOperator<Domain,Range>* coarseSolver_;
      bool isCoarsest_;
    };



    /**
     * @brief an algebraic multigrid method using a Krylov-cycle.
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

      enum {
        /** @brief The solver category. */
        category = Amg::category
      };
      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse
       * grid, must match the sparse matrix in the matrix hierachy.
       * @param smootherArgs The  arguments needed for thesmoother to use
       * for pre and post smoothing
       * @param gamma The number of subcycles. 1 for V-cycle, 2 for W-cycle.
       * @param preSmoothingSteps The number of smoothing steps for premoothing.
       * @param postSmoothingSteps The number of smoothing steps for postmoothing.
       * @param maxLevelKrylovSteps The maximum number of Krylov steps allowed at each level.
       * @param minDefectReduction The minimal defect reduction to achieve on each Krylov level.
       */
      KAMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
           const SmootherArgs& smootherArgs, std::size_t maxLevelKrylovSteps = 3 ,
           std::size_t preSmoothingSteps =1, std::size_t postSmoothingSteps = 1,
           double minDefectReduction =1e-1);

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
       * @param maxLevelKrylovSteps The maximum number of Krylov steps allowed at each level.
       * @param minDefectReduction The defect reduction to achieve on each krylov level.
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      KAMG(const Operator& fineOperator, const C& criterion,
           const SmootherArgs& smootherArgs, std::size_t maxLevelKrylovSteps=3,
           std::size_t preSmoothingSteps=1, std::size_t postSmoothingSteps=1,
           double minDefectReduction=1e-1,
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
      std::vector<typename Amg::ScalarProduct*> scalarproducts;

      /** @brief pointers to the allocated krylov solvers. */
      std::vector<KrylovSolver*> ksolvers;

      /** @brief pointers to the allocated krylov solvers. */
      std::vector<KAmgTwoGrid<Amg>*> twogridMethods;
    };

    template<class M, class X, class S, class P, class K, class A>
    KAMG<M,X,S,P,K,A>::KAMG(const OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
                            const SmootherArgs& smootherArgs,
                            std::size_t ksteps, std::size_t preSmoothingSteps,
                            std::size_t postSmoothingSteps,
                            double reduction)
      : amg(matrices, coarseSolver, smootherArgs, 1, preSmoothingSteps,
            postSmoothingSteps), maxLevelKrylovSteps(ksteps), levelDefectReduction(reduction)
    {}

    template<class M, class X, class S, class P, class K, class A>
    template<class C>
    KAMG<M,X,S,P,K,A>::KAMG(const Operator& fineOperator, const C& criterion,
                            const SmootherArgs& smootherArgs, std::size_t ksteps,
                            std::size_t preSmoothingSteps, std::size_t postSmoothingSteps,
                            double reduction,
                            const ParallelInformation& pinfo)
      : amg(fineOperator, criterion, smootherArgs, 1, preSmoothingSteps,
            postSmoothingSteps, false, pinfo), maxLevelKrylovSteps(ksteps), levelDefectReduction(reduction)
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

      if(hasCoarsest && matrix==amg.matrices_->matrices().finest())
        return;

      --matrix;
      --pinfo;
      twogridMethods.push_back(new KAmgTwoGrid<Amg>(amg, amg.solver_,true));
      ksolvers.push_back(new KrylovSolver(*matrix, *twogridMethods.back(),
                                          levelDefectReduction,maxLevelKrylovSteps,1));
      std::cout<<"push back true"<<std::endl;
      if(matrix!=amg.matrices_->matrices().finest())
        for(--matrix, --pinfo; true; --matrix, --pinfo) {
          twogridMethods.push_back(new KAmgTwoGrid<Amg>(amg, ksolvers.back(),false));
          ksolvers.push_back(new KrylovSolver(*matrix, *twogridMethods.back(),
                                              levelDefectReduction,maxLevelKrylovSteps,1));
          std::cout<<"push back false"<<std::endl;
          if(matrix==amg.matrices_->matrices().finest())
            break;
        }

      /*
         std::ostringstream s;

         while(true){
          scalarproducts.push_back(Amg::ScalarProductChooser::construct(*pinfo));
          ks = new KrylovSolver(*matrix, *(scalarproducts.back()),
         *(ksolvers.back()), levelDefectReduction,
                                maxLevelKrylovSteps, 0);
          if(matrix==amg.matrices_->matrices().finest())
            break;
          else{
            --matrix;
            --pinfo;
            ksolvers.push_front(new KAmgTwoGrid<Amg>(amg, ks));
            std::cout<<"push back false 2"<<std::endl;
          }
          }*/
    }


    template<class M, class X, class S, class P, class K, class A>
    void KAMG<M,X,S,P,K,A>::post(Domain& x)
    {
      typedef typename std::vector<KAmgTwoGrid<Amg>*>::reverse_iterator KIterator;
      typedef typename std::vector<ScalarProduct*>::iterator SIterator;
      for(KIterator kiter = twogridMethods.rbegin(); kiter != twogridMethods.rend();
          ++kiter)
      {
        if((*kiter)->coarseSolver()!=amg.solver_)
          delete (*kiter)->coarseSolver();
        delete *kiter;
      }
      for(SIterator siter = scalarproducts.begin(); siter!=scalarproducts.end();
          ++siter)
        delete *siter;
      amg.post(x);

    }

    template<class M, class X, class S, class P, class K, class A>
    void KAMG<M,X,S,P,K,A>::apply(Domain& v, const Range& d)
    {
      amg.initIteratorsWithFineLevel();
      *amg.lhs =v;
      *amg.rhs = d;
      *amg.update = 0;
      if(ksolvers.size()==0)
      {
        Range td=d;
        InverseOperatorResult res;
        amg.solver_->apply(v,td,res);
      }else{
        amg.presmooth();
        *amg.lhs=0;
#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION
        Range td=*amg.rhs;
        InverseOperatorResult res;
        ksolvers.back()->apply(*amg.lhs,td, res);
        *amg.update+=*amg.lhs;
#endif
        amg.postsmooth();
        v=*amg.update;
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
