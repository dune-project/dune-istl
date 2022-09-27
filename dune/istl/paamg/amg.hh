// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_AMG_HH
#define DUNE_AMG_AMG_HH

#include <memory>
#include <sstream>
#include <dune/common/exceptions.hh>
#include <dune/istl/paamg/smoother.hh>
#include <dune/istl/paamg/transfer.hh>
#include <dune/istl/paamg/matrixhierarchy.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/solvertype.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/scalarvectorview.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/parametertree.hh>

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

    template<class M, class X, class S, class P, class K, class A>
    class KAMG;

    template<class T>
    class KAmgTwoGrid;

    /**
     * @brief Parallel algebraic multigrid based on agglomeration.
     *
     * \tparam M The LinearOperator type which represents the matrix
     * \tparam X The vector type
     * \tparam S The smoother type
     * \tparam A An allocator for X
     *
     * \todo drop the smoother template parameter and replace with dynamic construction
     */
    template<class M, class X, class S, class PI=SequentialInformation,
        class A=std::allocator<X> >
    class AMG : public Preconditioner<X,X>
    {
      template<class M1, class X1, class S1, class P1, class K1, class A1>
      friend class KAMG;

      friend class KAmgTwoGrid<AMG>;

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

      /**
       * @brief Construct a new amg with a specific coarse solver.
       * @param matrices The already set up matix hierarchy.
       * @param coarseSolver The set up solver to use on the coarse
       * grid, must match the coarse matrix in the matrix hierarchy.
       * @param smootherArgs The  arguments needed for thesmoother to use
       * for pre and post smoothing.
       * @param parms The parameters for the AMG.
       */
      AMG(OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
          const SmootherArgs& smootherArgs, const Parameters& parms);

      /**
       * @brief Construct an AMG with an inexact coarse solver based on the smoother.
       *
       * As coarse solver a preconditioned CG method with the smoother as preconditioner
       * will be used. The matrix hierarchy is built automatically.
       * @param fineOperator The operator on the fine level.
       * @param criterion The criterion describing the coarsening strategy. E. g. SymmetricCriterion
       * or UnsymmetricCriterion, and providing the parameters.
       * @param smootherArgs The arguments for constructing the smoothers.
       * @param pinfo The information about the parallel distribution of the data.
       */
      template<class C>
      AMG(const Operator& fineOperator, const C& criterion,
          const SmootherArgs& smootherArgs=SmootherArgs(),
          const ParallelInformation& pinfo=ParallelInformation());

      /*!
         \brief Constructor an AMG via ParameterTree.

         \param fineOperator The operator on the fine level.
         \param configuration ParameterTree containing AMG parameters.
         \param pinfo Optionally, specify ParallelInformation

         ParameterTree Key         | Meaning
         --------------------------|------------
         smootherIterations        | The number of iterations to perform.
         smootherRelaxation        | The relaxation factor
         maxLevel                  | Maximum number of levels allowed in the hierarchy.
         coarsenTarget             | Maximum number of unknowns on the coarsest level.
         minCoarseningRate         | Coarsening will stop if the rate is below this threshold.
         accumulationMode          | If and how data is agglomerated on coarser level to
                                   | fewer processors. ("atOnce": do agglomeration once and
                                   | to one process; "successive": Multiple agglomerations to
                                   | fewer proceses until all data is on one process;
                                   | "none": Do no agglomeration at all and solve coarse level
                                   | iteratively).
         prolongationDampingFactor | Damping factor for the prolongation.
         alpha                     | Scaling avlue for marking connections as strong.
         beta                      | Treshold for marking nodes as isolated.
         additive                  | Whether to use additive multigrid.
         gamma                     | 1 for V-cycle, 2 for W-cycle.
         preSteps                  | Number of presmoothing steps.
         postSteps                 | Number of postsmoothing steps.
         verbosity                 | Output verbosity. default=2.
         criterionSymmetric        | If true use SymmetricCriterion (default), else UnSymmetricCriterion
         strengthMeasure           | What conversion to use to convert a matrix block
                                   | to a scalar when determining strength of connection:
                                   | diagonal (use a diagonal of row diagonalRowIndex, class Diagonal, default);
                                   | rowSum (rowSum norm), frobenius (Frobenius norm);
                                   | one (use always one and neglect the actual entries).
          diagonalRowIndex         | The index to use for the diagonal strength (default 0)
                                   | if this is i and strengthMeasure is "diagonal", then
                                   | block[i][i] will be used when determining strength of
                                   | connection.
          defaultAggregationSizeMode | Whether to set default values depending on isotropy of
                                   | problem uses parameters "defaultAggregationDimension" and
                                   | "maxAggregateDistance" (isotropic: For and isotropic problem;
                                   |  anisotropic: for an anisotropic problem).
          defaultAggregationDimension | Dimension of the problem (used for setting default aggregate size).
          maxAggregateDistance     | Maximum distance in an aggregte (in term of minimum edges needed to travel.
                                   | one vertex to another within the aggregate).
          minAggregateSize         | Minimum number of vertices an aggregate should consist of.
          maxAggregateSize         | Maximum number of vertices an aggregate should consist of.

         See \ref ISTL_Factory for the ParameterTree layout and examples.
       */
      AMG(std::shared_ptr<const Operator> fineOperator, const ParameterTree& configuration, const ParallelInformation& pinfo=ParallelInformation());

      /**
       * @brief Copy constructor.
       */
      AMG(const AMG& amg);

      /** \copydoc Preconditioner::pre */
      void pre(Domain& x, Range& b);

      /** \copydoc Preconditioner::apply */
      void apply(Domain& v, const Range& d);

      //! Category of the preconditioner (see SolverCategory::Category)
      virtual SolverCategory::Category category() const
      {
        return category_;
      }

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

      /**
       * @brief Recalculate the matrix hierarchy.
       *
       * It is assumed that the coarsening for the changed fine level
       * matrix would yield the same aggregates. In this case it suffices
       * to recalculate all the Galerkin products for the matrices of the
       * coarser levels.
       */
      void recalculateHierarchy()
      {
        matrices_->recalculateGalerkin(NegateSet<typename PI::OwnerSet>());
      }

      /**
       * @brief Check whether the coarse solver used is a direct solver.
       * @return True if the coarse level solver is a direct solver.
       */
      bool usesDirectCoarseLevelSolver() const;

    private:
      /*
       * @brief Helper function to create hierarchies with parameter tree.
       *
       * Will create the coarsen criterion with the norm and  create the
       * Hierarchies
       * \tparam Norm Type of the norm to use.
       */
      template<class Norm>
      void createCriterionAndHierarchies(std::shared_ptr<const Operator> matrixptr,
                                         const PI& pinfo, const Norm&,
                                         const ParameterTree& configuration,
                                         std::true_type compiles = std::true_type());
      template<class Norm>
      void createCriterionAndHierarchies(std::shared_ptr<const Operator> matrixptr,
                                         const PI& pinfo, const Norm&,
                                         const ParameterTree& configuration,
                                         std::false_type);
      /**
       * @brief Helper function to create hierarchies with settings from parameter tree.
       * @param criterion Coarsen criterion to configure and use
       */
      template<class C>
      void createHierarchies(C& criterion, std::shared_ptr<const Operator> matrixptr,
                             const PI& pinfo, const ParameterTree& configuration);
      /**
       * @brief Create matrix and smoother hierarchies.
       * @param criterion The coarsening criterion.
       * @param matrix The fine level matrix operator.
       * @param pinfo The fine level parallel information.
       */
      template<class C>
      void createHierarchies(C& criterion,
                             const std::shared_ptr<const Operator>& matrixptr,
                             const PI& pinfo);
      /**
       * @brief A struct that holds the context of the current level.
       *
       * These are the iterators to the smoother, matrix, parallel information,
       * and so on needed for the computations on the current level.
       */
      struct LevelContext
      {
        typedef Smoother SmootherType;
        /**
         * @brief The iterator over the smoothers.
         */
        typename Hierarchy<Smoother,A>::Iterator smoother;
        /**
         * @brief The iterator over the matrices.
         */
        typename OperatorHierarchy::ParallelMatrixHierarchy::ConstIterator matrix;
        /**
         * @brief The iterator over the parallel information.
         */
        typename ParallelInformationHierarchy::Iterator pinfo;
        /**
         * @brief The iterator over the redistribution information.
         */
        typename OperatorHierarchy::RedistributeInfoList::const_iterator redist;
        /**
         * @brief The iterator over the aggregates maps.
         */
        typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates;
        /**
         * @brief The iterator over the left hand side.
         */
        typename Hierarchy<Domain,A>::Iterator lhs;
        /**
         * @brief The iterator over the updates.
         */
        typename Hierarchy<Domain,A>::Iterator update;
        /**
         * @brief The iterator over the right hand sided.
         */
        typename Hierarchy<Range,A>::Iterator rhs;
        /**
         * @brief The level index.
         */
        std::size_t level;
      };


      /**
       * @brief Multigrid cycle on a level.
       * @param levelContext the iterators of the current level.
       */
      void mgc(LevelContext& levelContext);

      void additiveMgc();

      /**
       * @brief Move the iterators to the finer level
       * @param levelContext the iterators of the current level.
       * @param processedFineLevel Whether the process computed on
       *         fine level or not.
       */
      void moveToFineLevel(LevelContext& levelContext,bool processedFineLevel);

      /**
       * @brief Move the iterators to the coarser level.
       * @param levelContext the iterators of the current level
       */
      bool moveToCoarseLevel(LevelContext& levelContext);

      /**
       * @brief Initialize iterators over levels with fine level.
       * @param levelContext the iterators of the current level
       */
      void initIteratorsWithFineLevel(LevelContext& levelContext);

      /**  @brief The matrix we solve. */
      std::shared_ptr<OperatorHierarchy> matrices_;
      /** @brief The arguments to construct the smoother */
      SmootherArgs smootherArgs_;
      /** @brief The hierarchy of the smoothers. */
      std::shared_ptr<Hierarchy<Smoother,A> > smoothers_;
      /** @brief The solver of the coarsest level. */
      std::shared_ptr<CoarseSolver> solver_;
      /** @brief The right hand side of our problem. */
      std::shared_ptr<Hierarchy<Range,A>> rhs_;
      /** @brief The left approximate solution of our problem. */
      std::shared_ptr<Hierarchy<Domain,A>> lhs_;
      /** @brief The total update for the outer solver. */
      std::shared_ptr<Hierarchy<Domain,A>> update_;
      /** @brief The type of the scalar product for the coarse solver. */
      using ScalarProduct = Dune::ScalarProduct<X>;
      /** @brief Scalar product on the coarse level. */
      std::shared_ptr<ScalarProduct> scalarProduct_;
      /** @brief Gamma, 1 for V-cycle and 2 for W-cycle. */
      std::size_t gamma_;
      /** @brief The number of pre and postsmoothing steps. */
      std::size_t preSteps_;
      /** @brief The number of postsmoothing steps. */
      std::size_t postSteps_;
      bool buildHierarchy_;
      bool additive;
      bool coarsesolverconverged;
      std::shared_ptr<Smoother> coarseSmoother_;
      /** @brief The solver category. */
      SolverCategory::Category category_;
      /** @brief The verbosity level. */
      std::size_t verbosity_;

      struct ToLower
      {
        std::string operator()(const std::string& str)
        {
          std::stringstream retval;
          std::ostream_iterator<char> out(retval);
          std::transform(str.begin(), str.end(), out,
                         [](char c){
                           return std::tolower(c, std::locale::classic());
                         });
          return retval.str();
        }
      };
    };

    template<class M, class X, class S, class PI, class A>
    inline AMG<M,X,S,PI,A>::AMG(const AMG& amg)
    : matrices_(amg.matrices_), smootherArgs_(amg.smootherArgs_),
      smoothers_(amg.smoothers_), solver_(amg.solver_),
      rhs_(), lhs_(), update_(),
      scalarProduct_(amg.scalarProduct_), gamma_(amg.gamma_),
      preSteps_(amg.preSteps_), postSteps_(amg.postSteps_),
      buildHierarchy_(amg.buildHierarchy_),
      additive(amg.additive), coarsesolverconverged(amg.coarsesolverconverged),
      coarseSmoother_(amg.coarseSmoother_),
      category_(amg.category_),
      verbosity_(amg.verbosity_)
    {}

    template<class M, class X, class S, class PI, class A>
    AMG<M,X,S,PI,A>::AMG(OperatorHierarchy& matrices, CoarseSolver& coarseSolver,
                         const SmootherArgs& smootherArgs,
                         const Parameters& parms)
      : matrices_(stackobject_to_shared_ptr(matrices)), smootherArgs_(smootherArgs),
        smoothers_(new Hierarchy<Smoother,A>), solver_(&coarseSolver),
        rhs_(), lhs_(), update_(), scalarProduct_(0),
        gamma_(parms.getGamma()), preSteps_(parms.getNoPreSmoothSteps()),
        postSteps_(parms.getNoPostSmoothSteps()), buildHierarchy_(false),
        additive(parms.getAdditive()), coarsesolverconverged(true),
        coarseSmoother_(),
// #warning should category be retrieved from matrices?
        category_(SolverCategory::category(*smoothers_->coarsest())),
        verbosity_(parms.debugLevel())
    {
      assert(matrices_->isBuilt());

      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(*smoothers_, smootherArgs_);
    }

    template<class M, class X, class S, class PI, class A>
    template<class C>
    AMG<M,X,S,PI,A>::AMG(const Operator& matrix,
                         const C& criterion,
                         const SmootherArgs& smootherArgs,
                         const PI& pinfo)
      : smootherArgs_(smootherArgs),
        smoothers_(new Hierarchy<Smoother,A>), solver_(),
        rhs_(), lhs_(), update_(), scalarProduct_(),
        gamma_(criterion.getGamma()), preSteps_(criterion.getNoPreSmoothSteps()),
        postSteps_(criterion.getNoPostSmoothSteps()), buildHierarchy_(true),
        additive(criterion.getAdditive()), coarsesolverconverged(true),
        coarseSmoother_(),
        category_(SolverCategory::category(pinfo)),
        verbosity_(criterion.debugLevel())
    {
      if(SolverCategory::category(matrix) != SolverCategory::category(pinfo))
        DUNE_THROW(InvalidSolverCategory, "Matrix and Communication must have the same SolverCategory!");
      // TODO: reestablish compile time checks.
      //static_assert(static_cast<int>(PI::category)==static_cast<int>(S::category),
      //             "Matrix and Solver must match in terms of category!");
      auto matrixptr = stackobject_to_shared_ptr(matrix);
      createHierarchies(criterion, matrixptr, pinfo);
    }

    template<class M, class X, class S, class PI, class A>
    AMG<M,X,S,PI,A>::AMG(std::shared_ptr<const Operator> matrixptr,
                         const ParameterTree& configuration,
                         const ParallelInformation& pinfo) :
      smoothers_(new Hierarchy<Smoother,A>),
      solver_(), rhs_(), lhs_(), update_(), scalarProduct_(), buildHierarchy_(true),
      coarsesolverconverged(true), coarseSmoother_(),
      category_(SolverCategory::category(pinfo))
    {

      if (configuration.hasKey ("smootherIterations"))
        smootherArgs_.iterations = configuration.get<int>("smootherIterations");

      if (configuration.hasKey ("smootherRelaxation"))
        smootherArgs_.relaxationFactor = configuration.get<typename SmootherArgs::RelaxationFactor>("smootherRelaxation");

      auto normName = ToLower()(configuration.get("strengthMeasure", "diagonal"));
      auto index =  configuration.get<int>("diagonalRowIndex", 0);

      if ( normName == "diagonal")
      {
        using field_type = typename M::field_type;
        using real_type = typename FieldTraits<field_type>::real_type;
        std::is_convertible<field_type, real_type> compiles;

        switch (index)
        {
        case 0:
          createCriterionAndHierarchies(matrixptr, pinfo, Diagonal<0>(), configuration, compiles);
          break;
        case 1:
          createCriterionAndHierarchies(matrixptr, pinfo, Diagonal<1>(), configuration, compiles);
          break;
        case 2:
          createCriterionAndHierarchies(matrixptr, pinfo, Diagonal<2>(), configuration, compiles);
          break;
        case 3:
          createCriterionAndHierarchies(matrixptr, pinfo, Diagonal<3>(), configuration, compiles);
          break;
        case 4:
          createCriterionAndHierarchies(matrixptr, pinfo, Diagonal<4>(), configuration, compiles);
          break;
        default:
          DUNE_THROW(InvalidStateException, "Currently strengthIndex>4 is not supported.");
        }
      }
      else if (normName == "rowsum")
        createCriterionAndHierarchies(matrixptr, pinfo, RowSum(), configuration);
      else if (normName == "frobenius")
        createCriterionAndHierarchies(matrixptr, pinfo, FrobeniusNorm(), configuration);
      else if (normName == "one")
        createCriterionAndHierarchies(matrixptr, pinfo, AlwaysOneNorm(), configuration);
      else
        DUNE_THROW(Dune::NotImplemented, "Wrong config file: strengthMeasure "<<normName<<" is not supported by AMG");
    }

  template<class M, class X, class S, class PI, class A>
  template<class Norm>
  void AMG<M,X,S,PI,A>::createCriterionAndHierarchies(std::shared_ptr<const Operator> matrixptr, const PI& pinfo, const Norm&, const ParameterTree& configuration, std::false_type)
  {
    DUNE_THROW(InvalidStateException, "Strength of connection measure does not support this type ("
               << className<typename M::field_type>() << ") as it is lacking a conversion to"
               << className<typename FieldTraits<typename M::field_type>::real_type>() << ".");
  }

  template<class M, class X, class S, class PI, class A>
  template<class Norm>
  void AMG<M,X,S,PI,A>::createCriterionAndHierarchies(std::shared_ptr<const Operator> matrixptr, const PI& pinfo, const Norm&, const ParameterTree& configuration, std::true_type)
  {
    if (configuration.get<bool>("criterionSymmetric", true))
      {
        using Criterion = Dune::Amg::CoarsenCriterion<
          Dune::Amg::SymmetricCriterion<typename M::matrix_type,Norm> >;
        Criterion criterion;
        createHierarchies(criterion, matrixptr, pinfo, configuration);
      }
      else
      {
        using Criterion = Dune::Amg::CoarsenCriterion<
          Dune::Amg::UnSymmetricCriterion<typename M::matrix_type,Norm> >;
        Criterion criterion;
        createHierarchies(criterion, matrixptr, pinfo, configuration);
      }
    }

  template<class M, class X, class S, class PI, class A>
  template<class C>
  void AMG<M,X,S,PI,A>::createHierarchies(C& criterion, std::shared_ptr<const Operator> matrixptr, const PI& pinfo, const ParameterTree& configuration)
  {
      if (configuration.hasKey ("maxLevel"))
        criterion.setMaxLevel(configuration.get<int>("maxLevel"));

      if (configuration.hasKey ("minCoarseningRate"))
        criterion.setMinCoarsenRate(configuration.get<int>("minCoarseningRate"));

      if (configuration.hasKey ("coarsenTarget"))
        criterion.setCoarsenTarget (configuration.get<int>("coarsenTarget"));

      if (configuration.hasKey ("accumulationMode"))
      {
        std::string mode = ToLower()(configuration.get<std::string>("accumulationMode"));
        if ( mode == "none")
          criterion.setAccumulate(AccumulationMode::noAccu);
        else if ( mode == "atonce" )
          criterion.setAccumulate(AccumulationMode::atOnceAccu);
        else if ( mode == "successive")
          criterion.setCoarsenTarget (AccumulationMode::successiveAccu);
        else
          DUNE_THROW(InvalidSolverFactoryConfiguration, "Parameter accumulationMode does not allow value "
                     << mode <<".");
      }

      if (configuration.hasKey ("prolongationDampingFactor"))
        criterion.setProlongationDampingFactor (configuration.get<double>("prolongationDampingFactor"));

      if (configuration.hasKey("defaultAggregationSizeMode"))
      {
        auto mode = ToLower()(configuration.get<std::string>("defaultAggregationSizeMode"));
        auto dim = configuration.get<std::size_t>("defaultAggregationDimension");
        std::size_t maxDistance = 2;
        if (configuration.hasKey("MaxAggregateDistance"))
          maxDistance = configuration.get<std::size_t>("maxAggregateDistance");
        if (mode == "isotropic")
          criterion.setDefaultValuesIsotropic(dim, maxDistance);
        else if(mode == "anisotropic")
          criterion.setDefaultValuesAnisotropic(dim, maxDistance);
        else
          DUNE_THROW(InvalidSolverFactoryConfiguration, "Parameter accumulationMode does not allow value "
                   << mode <<".");
      }

      if (configuration.hasKey("maxAggregateDistance"))
        criterion.setMaxDistance(configuration.get<std::size_t>("maxAggregateDistance"));

      if (configuration.hasKey("minAggregateSize"))
        criterion.setMinAggregateSize(configuration.get<std::size_t>("minAggregateSize"));

      if (configuration.hasKey("maxAggregateSize"))
        criterion.setMaxAggregateSize(configuration.get<std::size_t>("maxAggregateSize"));

      if (configuration.hasKey("maxAggregateConnectivity"))
        criterion.setMaxConnectivity(configuration.get<std::size_t>("maxAggregateConnectivity"));

      if (configuration.hasKey ("alpha"))
        criterion.setAlpha (configuration.get<double> ("alpha"));

      if (configuration.hasKey ("beta"))
        criterion.setBeta (configuration.get<double> ("beta"));

      if (configuration.hasKey ("gamma"))
        criterion.setGamma (configuration.get<std::size_t> ("gamma"));
      gamma_ = criterion.getGamma();

      if (configuration.hasKey ("additive"))
        criterion.setAdditive (configuration.get<bool>("additive"));
      additive = criterion.getAdditive();

      if (configuration.hasKey ("preSteps"))
        criterion.setNoPreSmoothSteps (configuration.get<std::size_t> ("preSteps"));
      preSteps_ = criterion.getNoPreSmoothSteps ();

      if (configuration.hasKey ("postSteps"))
        criterion.setNoPostSmoothSteps (configuration.get<std::size_t> ("postSteps"));
      postSteps_ = criterion.getNoPostSmoothSteps ();

      verbosity_ = configuration.get("verbosity", 0);
      criterion.setDebugLevel (verbosity_);

      createHierarchies(criterion, matrixptr, pinfo);
    }

    template <class Matrix,
              class Vector>
    struct DirectSolverSelector
    {
      typedef typename Matrix :: field_type field_type;
      enum SolverType { umfpack, superlu, none };

      static constexpr SolverType solver =
#if DISABLE_AMG_DIRECTSOLVER
        none;
#elif HAVE_SUITESPARSE_UMFPACK
        UMFPackMethodChooser< field_type > :: valid ? umfpack : none ;
#elif HAVE_SUPERLU
        superlu ;
#else
        none;
#endif

      template <class M, SolverType>
      struct Solver
      {
        typedef InverseOperator<Vector,Vector> type;
        static type* create(const M& mat, bool verbose, bool reusevector )
        {
          DUNE_THROW(NotImplemented,"DirectSolver not selected");
          return nullptr;
        }
        static std::string name () { return "None"; }
      };
#if HAVE_SUITESPARSE_UMFPACK
      template <class M>
      struct Solver< M, umfpack >
      {
        typedef UMFPack< M > type;
        static type* create(const M& mat, bool verbose, bool reusevector )
        {
          return new type(mat, verbose, reusevector );
        }
        static std::string name () { return "UMFPack"; }
      };
#endif
#if HAVE_SUPERLU
      template <class M>
      struct Solver< M, superlu >
      {
        typedef SuperLU< M > type;
        static type* create(const M& mat, bool verbose, bool reusevector )
        {
          return new type(mat, verbose, reusevector );
        }
        static std::string name () { return "SuperLU"; }
      };
#endif

      // define direct solver type to be used
      typedef Solver< Matrix, solver > SelectedSolver ;
      typedef typename SelectedSolver :: type   DirectSolver;
      static constexpr bool isDirectSolver = solver != none;
      static std::string name() { return SelectedSolver :: name (); }
      static DirectSolver* create(const Matrix& mat, bool verbose, bool reusevector )
      {
        return SelectedSolver :: create( mat, verbose, reusevector );
      }
    };

    template<class M, class X, class S, class PI, class A>
    template<class C>
    void AMG<M,X,S,PI,A>::createHierarchies(C& criterion,
      const std::shared_ptr<const Operator>& matrixptr,
                                            const PI& pinfo)
    {
      Timer watch;
      matrices_ = std::make_shared<OperatorHierarchy>(
        std::const_pointer_cast<Operator>(matrixptr),
        stackobject_to_shared_ptr(const_cast<PI&>(pinfo)));

      matrices_->template build<NegateSet<typename PI::OwnerSet> >(criterion);

      // build the necessary smoother hierarchies
      matrices_->coarsenSmoother(*smoothers_, smootherArgs_);

      // test whether we should solve on the coarse level. That is the case if we
      // have that level and if there was a redistribution on this level then our
      // communicator has to be valid (size()>0) as the smoother might try to communicate
      // in the constructor.
      if(buildHierarchy_ && matrices_->levels()==matrices_->maxlevels()
         && ( ! matrices_->redistributeInformation().back().isSetup() ||
              matrices_->parallelInformation().coarsest().getRedistributed().communicator().size() ) )
      {
        // We have the carsest level. Create the coarse Solver
        SmootherArgs sargs(smootherArgs_);
        sargs.iterations = 1;

        typename ConstructionTraits<Smoother>::Arguments cargs;
        cargs.setArgs(sargs);
        if(matrices_->redistributeInformation().back().isSetup()) {
          // Solve on the redistributed partitioning
          cargs.setMatrix(matrices_->matrices().coarsest().getRedistributed().getmat());
          cargs.setComm(matrices_->parallelInformation().coarsest().getRedistributed());
        }else{
          cargs.setMatrix(matrices_->matrices().coarsest()->getmat());
          cargs.setComm(*matrices_->parallelInformation().coarsest());
        }

        coarseSmoother_ = ConstructionTraits<Smoother>::construct(cargs);
        scalarProduct_ = createScalarProduct<X>(cargs.getComm(),category());

        typedef DirectSolverSelector< typename M::matrix_type, X > SolverSelector;

        // Use superlu if we are purely sequential or with only one processor on the coarsest level.
        if( SolverSelector::isDirectSolver &&
            (std::is_same<ParallelInformation,SequentialInformation>::value // sequential mode
           || matrices_->parallelInformation().coarsest()->communicator().size()==1 //parallel mode and only one processor
           || (matrices_->parallelInformation().coarsest().isRedistributed()
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()==1
               && matrices_->parallelInformation().coarsest().getRedistributed().communicator().size()>0) )
          )
        { // redistribute and 1 proc
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
            {
              // We are still participating on this level
              solver_.reset(SolverSelector::create(matrices_->matrices().coarsest().getRedistributed().getmat(), false, false));
            }
            else
              solver_.reset();
          }
          else
          {
            solver_.reset(SolverSelector::create(matrices_->matrices().coarsest()->getmat(), false, false));
          }
          if(verbosity_>0 && matrices_->parallelInformation().coarsest()->communicator().rank()==0)
            std::cout<< "Using a direct coarse solver (" << SolverSelector::name() << ")" << std::endl;
        }
        else
        {
          if(matrices_->parallelInformation().coarsest().isRedistributed())
          {
            if(matrices_->matrices().coarsest().getRedistributed().getmat().N()>0)
              // We are still participating on this level

              // we have to allocate these types using the rebound allocator
              // in order to ensure that we fulfill the alignment requirements
              solver_.reset(new BiCGSTABSolver<X>(const_cast<M&>(matrices_->matrices().coarsest().getRedistributed()),
                                                  *scalarProduct_,
                                                  *coarseSmoother_, 1E-2, 1000, 0));
            else
              solver_.reset();
          }else
          {
              solver_.reset(new BiCGSTABSolver<X>(const_cast<M&>(*matrices_->matrices().coarsest()),
                  *scalarProduct_,
                  *coarseSmoother_, 1E-2, 1000, 0));
            // // we have to allocate these types using the rebound allocator
            // // in order to ensure that we fulfill the alignment requirements
            // using Alloc = typename std::allocator_traits<A>::template rebind_alloc<BiCGSTABSolver<X>>;
            // Alloc alloc;
            // auto p = alloc.allocate(1);
            // std::allocator_traits<Alloc>::construct(alloc, p,
            //   const_cast<M&>(*matrices_->matrices().coarsest()),
            //   *scalarProduct_,
            //   *coarseSmoother_, 1E-2, 1000, 0);
            // solver_.reset(p,[](BiCGSTABSolver<X>* p){
            //     Alloc alloc;
            //     std::allocator_traits<Alloc>::destroy(alloc, p);
            //     alloc.deallocate(p,1);
            //   });
          }
        }
      }

      if(verbosity_>0 && matrices_->parallelInformation().finest()->communicator().rank()==0)
        std::cout<<"Building hierarchy of "<<matrices_->maxlevels()<<" levels "
                 <<"(including coarse solver) took "<<watch.elapsed()<<" seconds."<<std::endl;
    }


    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::pre(Domain& x, Range& b)
    {
      // Detect Matrix rows where all offdiagonal entries are
      // zero and set x such that  A_dd*x_d=b_d
      // Thus users can be more careless when setting up their linear
      // systems.
      typedef typename M::matrix_type Matrix;
      typedef typename Matrix::ConstRowIterator RowIter;
      typedef typename Matrix::ConstColIterator ColIter;
      typedef typename Matrix::block_type Block;
      Block zero;
      zero=typename Matrix::field_type();

      const Matrix& mat=matrices_->matrices().finest()->getmat();
      for(RowIter row=mat.begin(); row!=mat.end(); ++row) {
        bool isDirichlet = true;
        bool hasDiagonal = false;
        Block diagonal{};
        for(ColIter col=row->begin(); col!=row->end(); ++col) {
          if(row.index()==col.index()) {
            diagonal = *col;
            hasDiagonal = true;
          }else{
            if(*col!=zero)
              isDirichlet = false;
          }
        }
        if(isDirichlet && hasDiagonal)
        {
          auto&& xEntry = Impl::asVector(x[row.index()]);
          auto&& bEntry = Impl::asVector(b[row.index()]);
          Impl::asMatrix(diagonal).solve(xEntry, bEntry);
        }
      }

      if(smoothers_->levels()>0)
        smoothers_->finest()->pre(x,b);
      else
        // No smoother to make x consistent! Do it by hand
        matrices_->parallelInformation().coarsest()->copyOwnerToAll(x,x);
      rhs_ = std::make_shared<Hierarchy<Range,A>>(std::make_shared<Range>(b));
      lhs_ = std::make_shared<Hierarchy<Domain,A>>(std::make_shared<Domain>(x));
      update_ = std::make_shared<Hierarchy<Domain,A>>(std::make_shared<Domain>(x));
      matrices_->coarsenVector(*rhs_);
      matrices_->coarsenVector(*lhs_);
      matrices_->coarsenVector(*update_);

      // Preprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Range,A>::Iterator RIterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_->coarsest();
      Iterator smoother = smoothers_->finest();
      RIterator rhs = rhs_->finest();
      DIterator lhs = lhs_->finest();
      if(smoothers_->levels()>1) {

        assert(lhs_->levels()==rhs_->levels());
        assert(smoothers_->levels()==lhs_->levels() || matrices_->levels()==matrices_->maxlevels());
        assert(smoothers_->levels()+1==lhs_->levels() || matrices_->levels()<matrices_->maxlevels());

        if(smoother!=coarsest)
          for(++smoother, ++lhs, ++rhs; smoother != coarsest; ++smoother, ++lhs, ++rhs)
            smoother->pre(*lhs,*rhs);
        smoother->pre(*lhs,*rhs);
      }


      // The preconditioner might change x and b. So we have to
      // copy the changes to the original vectors.
      x = *lhs_->finest();
      b = *rhs_->finest();

    }
    template<class M, class X, class S, class PI, class A>
    std::size_t AMG<M,X,S,PI,A>::levels()
    {
      return matrices_->levels();
    }
    template<class M, class X, class S, class PI, class A>
    std::size_t AMG<M,X,S,PI,A>::maxlevels()
    {
      return matrices_->maxlevels();
    }

    /** \copydoc Preconditioner::apply */
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::apply(Domain& v, const Range& d)
    {
      LevelContext levelContext;

      if(additive) {
        *(rhs_->finest())=d;
        additiveMgc();
        v=*lhs_->finest();
      }else{
        // Init all iterators for the current level
        initIteratorsWithFineLevel(levelContext);


        *levelContext.lhs = v;
        *levelContext.rhs = d;
        *levelContext.update=0;
        levelContext.level=0;

        mgc(levelContext);

        if(postSteps_==0||matrices_->maxlevels()==1)
          levelContext.pinfo->copyOwnerToAll(*levelContext.update, *levelContext.update);

        v=*levelContext.update;
      }

    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::initIteratorsWithFineLevel(LevelContext& levelContext)
    {
      levelContext.smoother = smoothers_->finest();
      levelContext.matrix = matrices_->matrices().finest();
      levelContext.pinfo = matrices_->parallelInformation().finest();
      levelContext.redist =
        matrices_->redistributeInformation().begin();
      levelContext.aggregates = matrices_->aggregatesMaps().begin();
      levelContext.lhs = lhs_->finest();
      levelContext.update = update_->finest();
      levelContext.rhs = rhs_->finest();
    }

    template<class M, class X, class S, class PI, class A>
    bool AMG<M,X,S,PI,A>
    ::moveToCoarseLevel(LevelContext& levelContext)
    {

      bool processNextLevel=true;

      if(levelContext.redist->isSetup()) {
        levelContext.redist->redistribute(static_cast<const Range&>(*levelContext.rhs),
                             levelContext.rhs.getRedistributed());
        processNextLevel = levelContext.rhs.getRedistributed().size()>0;
        if(processNextLevel) {
          //restrict defect to coarse level right hand side.
          typename Hierarchy<Range,A>::Iterator fineRhs = levelContext.rhs++;
          ++levelContext.pinfo;
          Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
          ::restrictVector(*(*levelContext.aggregates), *levelContext.rhs,
                           static_cast<const Range&>(fineRhs.getRedistributed()),
                           *levelContext.pinfo);
        }
      }else{
        //restrict defect to coarse level right hand side.
        typename Hierarchy<Range,A>::Iterator fineRhs = levelContext.rhs++;
        ++levelContext.pinfo;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::restrictVector(*(*levelContext.aggregates),
                         *levelContext.rhs, static_cast<const Range&>(*fineRhs),
                         *levelContext.pinfo);
      }

      if(processNextLevel) {
        // prepare coarse system
        ++levelContext.lhs;
        ++levelContext.update;
        ++levelContext.matrix;
        ++levelContext.level;
        ++levelContext.redist;

        if(levelContext.matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
          // next level is not the globally coarsest one
          ++levelContext.smoother;
          ++levelContext.aggregates;
        }
        // prepare the update on the next level
        *levelContext.update=0;
      }
      return processNextLevel;
    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>
    ::moveToFineLevel(LevelContext& levelContext, bool processNextLevel)
    {
      if(processNextLevel) {
        if(levelContext.matrix != matrices_->matrices().coarsest() || matrices_->levels()<matrices_->maxlevels()) {
          // previous level is not the globally coarsest one
          --levelContext.smoother;
          --levelContext.aggregates;
        }
        --levelContext.redist;
        --levelContext.level;
        //prolongate and add the correction (update is in coarse left hand side)
        --levelContext.matrix;

        //typename Hierarchy<Domain,A>::Iterator coarseLhs = lhs--;
        --levelContext.lhs;
        --levelContext.pinfo;
      }
      if(levelContext.redist->isSetup()) {
        // Need to redistribute during prolongateVector
        levelContext.lhs.getRedistributed()=0;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::prolongateVector(*(*levelContext.aggregates), *levelContext.update, *levelContext.lhs,
                           levelContext.lhs.getRedistributed(),
                           matrices_->getProlongationDampingFactor(),
                           *levelContext.pinfo, *levelContext.redist);
      }else{
        *levelContext.lhs=0;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::prolongateVector(*(*levelContext.aggregates), *levelContext.update, *levelContext.lhs,
                           matrices_->getProlongationDampingFactor(),
                           *levelContext.pinfo);
      }


      if(processNextLevel) {
        --levelContext.update;
        --levelContext.rhs;
      }

      *levelContext.update += *levelContext.lhs;
    }

    template<class M, class X, class S, class PI, class A>
    bool AMG<M,X,S,PI,A>::usesDirectCoarseLevelSolver() const
    {
      return IsDirectSolver< CoarseSolver>::value;
    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::mgc(LevelContext& levelContext){
      if(levelContext.matrix == matrices_->matrices().coarsest() && levels()==maxlevels()) {
        // Solve directly
        InverseOperatorResult res;
        res.converged=true; // If we do not compute this flag will not get updated
        if(levelContext.redist->isSetup()) {
          levelContext.redist->redistribute(*levelContext.rhs, levelContext.rhs.getRedistributed());
          if(levelContext.rhs.getRedistributed().size()>0) {
            // We are still participating in the computation
            levelContext.pinfo.getRedistributed().copyOwnerToAll(levelContext.rhs.getRedistributed(),
                                                    levelContext.rhs.getRedistributed());
            solver_->apply(levelContext.update.getRedistributed(),
                           levelContext.rhs.getRedistributed(), res);
          }
          levelContext.redist->redistributeBackward(*levelContext.update, levelContext.update.getRedistributed());
          levelContext.pinfo->copyOwnerToAll(*levelContext.update, *levelContext.update);
        }else{
          levelContext.pinfo->copyOwnerToAll(*levelContext.rhs, *levelContext.rhs);
          solver_->apply(*levelContext.update, *levelContext.rhs, res);
        }

        if (!res.converged)
          coarsesolverconverged = false;
      }else{
        // presmoothing
        presmooth(levelContext, preSteps_);

#ifndef DUNE_AMG_NO_COARSEGRIDCORRECTION
        bool processNextLevel = moveToCoarseLevel(levelContext);

        if(processNextLevel) {
          // next level
          for(std::size_t i=0; i<gamma_; i++){
            mgc(levelContext);
            if (levelContext.matrix == matrices_->matrices().coarsest() && levels()==maxlevels())
              break;
            if(i+1 < gamma_){
              levelContext.matrix->applyscaleadd(-1., *levelContext.lhs, *levelContext.rhs);
            }
          }
        }

        moveToFineLevel(levelContext, processNextLevel);
#else
        *lhs=0;
#endif

        if(levelContext.matrix == matrices_->matrices().finest()) {
          coarsesolverconverged = matrices_->parallelInformation().finest()->communicator().prod(coarsesolverconverged);
          if(!coarsesolverconverged)
            DUNE_THROW(MathError, "Coarse solver did not converge");
        }
        // postsmoothing
        postsmooth(levelContext, postSteps_);

      }
    }

    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::additiveMgc(){

      // restrict residual to all levels
      typename ParallelInformationHierarchy::Iterator pinfo=matrices_->parallelInformation().finest();
      typename Hierarchy<Range,A>::Iterator rhs=rhs_->finest();
      typename Hierarchy<Domain,A>::Iterator lhs = lhs_->finest();
      typename OperatorHierarchy::AggregatesMapList::const_iterator aggregates=matrices_->aggregatesMaps().begin();

      for(typename Hierarchy<Range,A>::Iterator fineRhs=rhs++; fineRhs != rhs_->coarsest(); fineRhs=rhs++, ++aggregates) {
        ++pinfo;
        Transfer<typename OperatorHierarchy::AggregatesMap::AggregateDescriptor,Range,ParallelInformation>
        ::restrictVector(*(*aggregates), *rhs, static_cast<const Range&>(*fineRhs), *pinfo);
      }

      // pinfo is invalid, set to coarsest level
      //pinfo = matrices_->parallelInformation().coarsest
      // calculate correction for all levels
      lhs = lhs_->finest();
      typename Hierarchy<Smoother,A>::Iterator smoother = smoothers_->finest();

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
        ::prolongateVector(*(*aggregates), *coarseLhs, *lhs, 1.0, *pinfo);
      }
    }


    /** \copydoc Preconditioner::post */
    template<class M, class X, class S, class PI, class A>
    void AMG<M,X,S,PI,A>::post([[maybe_unused]] Domain& x)
    {
      // Postprocess all smoothers
      typedef typename Hierarchy<Smoother,A>::Iterator Iterator;
      typedef typename Hierarchy<Domain,A>::Iterator DIterator;
      Iterator coarsest = smoothers_->coarsest();
      Iterator smoother = smoothers_->finest();
      DIterator lhs = lhs_->finest();
      if(smoothers_->levels()>0) {
        if(smoother != coarsest  || matrices_->levels()<matrices_->maxlevels())
          smoother->post(*lhs);
        if(smoother!=coarsest)
          for(++smoother, ++lhs; smoother != coarsest; ++smoother, ++lhs)
            smoother->post(*lhs);
        smoother->post(*lhs);
      }
      lhs_ = nullptr;
      update_ = nullptr;
      rhs_ = nullptr;
    }

    template<class M, class X, class S, class PI, class A>
    template<class A1>
    void AMG<M,X,S,PI,A>::getCoarsestAggregateNumbers(std::vector<std::size_t,A1>& cont)
    {
      matrices_->getCoarsestAggregatesOnFinest(cont);
    }

  } // end namespace Amg

  struct AMGCreator{
    template<class> struct isValidBlockType : std::false_type{};
    template<class T, int n, int m> struct isValidBlockType<FieldMatrix<T,n,m>> : std::true_type{};

    template<class OP>
    std::shared_ptr<Dune::Preconditioner<typename OP::element_type::domain_type, typename OP::element_type::range_type> >
    makeAMG(const OP& op, const std::string& smoother, const Dune::ParameterTree& config) const
    {
      DUNE_THROW(Dune::Exception, "Operator type not supported by AMG");
    }

    template<class M, class X, class Y>
    std::shared_ptr<Dune::Preconditioner<X,Y> >
    makeAMG(const std::shared_ptr<MatrixAdapter<M,X,Y>>& op, const std::string& smoother,
            const Dune::ParameterTree& config) const
    {
      using OP = MatrixAdapter<M,X,Y>;

      if(smoother == "ssor")
        return std::make_shared<Amg::AMG<OP, X, SeqSSOR<M,X,Y>>>(op, config);
      if(smoother == "sor")
        return std::make_shared<Amg::AMG<OP, X, SeqSOR<M,X,Y>>>(op, config);
      if(smoother == "jac")
        return std::make_shared<Amg::AMG<OP, X, SeqJac<M,X,Y>>>(op, config);
      if(smoother == "gs")
        return std::make_shared<Amg::AMG<OP, X, SeqGS<M,X,Y>>>(op, config);
      if(smoother == "ilu")
        return std::make_shared<Amg::AMG<OP, X, SeqILU<M,X,Y>>>(op, config);
      else
        DUNE_THROW(Dune::Exception, "Unknown smoother for AMG");
    }

    template<class M, class X, class Y, class C>
    std::shared_ptr<Dune::Preconditioner<X,Y> >
    makeAMG(const std::shared_ptr<OverlappingSchwarzOperator<M,X,Y,C>>& op, const std::string& smoother,
            const Dune::ParameterTree& config) const
    {
      using OP = OverlappingSchwarzOperator<M,X,Y,C>;

      auto cop = std::static_pointer_cast<const OP>(op);

      if(smoother == "ssor")
        return std::make_shared<Amg::AMG<OP, X, BlockPreconditioner<X,Y,C,SeqSSOR<M,X,Y>>,C>>(cop, config, op->getCommunication());
      if(smoother == "sor")
        return std::make_shared<Amg::AMG<OP, X, BlockPreconditioner<X,Y,C,SeqSOR<M,X,Y>>,C>>(cop, config, op->getCommunication());
      if(smoother == "jac")
        return std::make_shared<Amg::AMG<OP, X, BlockPreconditioner<X,Y,C,SeqJac<M,X,Y>>,C>>(cop, config, op->getCommunication());
      if(smoother == "gs")
        return std::make_shared<Amg::AMG<OP, X, BlockPreconditioner<X,Y,C,SeqGS<M,X,Y>>,C>>(cop, config, op->getCommunication());
      if(smoother == "ilu")
        return std::make_shared<Amg::AMG<OP, X, BlockPreconditioner<X,Y,C,SeqILU<M,X,Y>>,C>>(cop, config, op->getCommunication());
      else
        DUNE_THROW(Dune::Exception, "Unknown smoother for AMG");
    }

    template<class M, class X, class Y, class C>
    std::shared_ptr<Dune::Preconditioner<X,Y> >
    makeAMG(const std::shared_ptr<NonoverlappingSchwarzOperator<M,X,Y,C>>& op, const std::string& smoother,
            const Dune::ParameterTree& config) const
    {
      using OP = NonoverlappingSchwarzOperator<M,X,Y,C>;

      if(smoother == "ssor")
        return std::make_shared<Amg::AMG<OP, X, NonoverlappingBlockPreconditioner<C,SeqSSOR<M,X,Y>>,C>>(op, config, op->getCommunication());
      if(smoother == "sor")
        return std::make_shared<Amg::AMG<OP, X, NonoverlappingBlockPreconditioner<C,SeqSOR<M,X,Y>>,C>>(op, config, op->getCommunication());
      if(smoother == "jac")
        return std::make_shared<Amg::AMG<OP, X, NonoverlappingBlockPreconditioner<C,SeqJac<M,X,Y>>,C>>(op, config, op->getCommunication());
      if(smoother == "gs")
        return std::make_shared<Amg::AMG<OP, X, NonoverlappingBlockPreconditioner<C,SeqGS<M,X,Y>>,C>>(op, config, op->getCommunication());
      if(smoother == "ilu")
        return std::make_shared<Amg::AMG<OP, X, NonoverlappingBlockPreconditioner<C,SeqILU<M,X,Y>>,C>>(op, config, op->getCommunication());
      else
        DUNE_THROW(Dune::Exception, "Unknown smoother for AMG");
    }

    template<typename TL, typename OP>
    std::shared_ptr<Dune::Preconditioner<typename Dune::TypeListElement<1, TL>::type,
                                         typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL tl, const std::shared_ptr<OP>& op, const Dune::ParameterTree& config,
                std::enable_if_t<isValidBlockType<typename OP::matrix_type::block_type>::value,int> = 0) const
    {
      using field_type = typename OP::matrix_type::field_type;
      using real_type = typename FieldTraits<field_type>::real_type;
      if (!std::is_convertible<field_type, real_type>())
        DUNE_THROW(UnsupportedType, "AMG needs field_type(" <<
                   className<field_type>() <<
                   ") to be convertible to its real_type (" <<
                   className<real_type>() <<
                   ").");
      using D = typename Dune::TypeListElement<1, decltype(tl)>::type;
      using R = typename Dune::TypeListElement<2, decltype(tl)>::type;
      std::shared_ptr<Preconditioner<D,R>> amg;
      std::string smoother = config.get("smoother", "ssor");
      return makeAMG(op, smoother, config);
    }

    template<typename TL, typename OP>
    std::shared_ptr<Dune::Preconditioner<typename Dune::TypeListElement<1, TL>::type,
                                         typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const std::shared_ptr<OP>& /*mat*/, const Dune::ParameterTree& /*config*/,
                std::enable_if_t<!isValidBlockType<typename OP::matrix_type::block_type>::value,int> = 0) const
    {
      DUNE_THROW(UnsupportedType, "AMG needs a FieldMatrix as Matrix block_type");
    }
  };

  DUNE_REGISTER_PRECONDITIONER("amg", AMGCreator());
} // end namespace Dune

#endif
