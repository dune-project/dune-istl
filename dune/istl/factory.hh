// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FACTORY_HH
#define DUNE_ISTL_FACTORY_HH

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/paamg/amg.hh>


/**
 *  @defgroup ISTL_Factory ISTL Factory
 *  These factories allow creating preconditioners and solvers from ParameterTrees.
 *
 *  The ParameterTree Layout
 *  ========================
 *
 *  This is a sample ini file for creating a solver:
 *
 *      [solver]
 *      precond = SeqSSOR
 *      solver = CGSolver
 *
 *      [SeqSSOR]
 *      iterations = 1
 *      relaxation = 1.8
 *
 *      [CGSolver]
 *      reduction=1e-9
 *      maxit = 1000
 *      verbose = 3
 *
 *  \note By default, the preconditioners / solvers are identified by their class names.
 *
 *  The file can be read by your program like this:
 *
 *      // Read config.ini
 *      Dune::const ParameterTree& configuration;
 *      Dune::ParameterTreeParser parser;
 *      parser.readINITree("config.ini", configuration);
 *
 *  Configuration Parameters
 *  ==================================
 *  Each preconditioner or solver can have specific parameters. You can find these in their respective documentations.
 *
 *  Common Parameters {#ISTL_Factory_Common_Params}
 *  -----------------
 *
 *  These parameters are common for preconditioners:
 *
 *     ParameterTree Key | Meaning
 *     ------------------|------------
 *     relaxation        | The relaxation factor.
 *
 *  And these parameters are common for solvers:
 *
 *     ParameterTree Key | Meaning
 *     ------------------|------------
 *     maxit             | The maximum number of iteration steps allowed when applying the operator.
 *     reduction         | The relative defect reduction to achieve when applying the operator.
 *     verbose           | Verbose levels are:<ul><li>0: print nothing</li><li>1: print initial and final defect and statistics</li><li>2: print line for each iteration</li></ul>
 *
 *  Multiple Components of Same Type
 *  --------------------------------
 *
 *  If you need multiple configurations for the same preconditioner or solver type, you can name them arbitrarily and
 *  specify their type explicitly.
 *
 *  In the example above, this leads to:
 *
 *      [solver]
 *      precond = SeqSSOR
 *      solver = StubbornCGSolver
 *
 *      [SeqSSOR]
 *      iterations = 1
 *      relaxation = 1.8
 *
 *      [CGSolver]
 *      reduction=1e-9
 *      maxit = 1000
 *      verbose = 3
 *
 *      [StubbornCGSolver]
 *      type = CGSolver
 *      reduction=1e-9
 *      maxit = 50000
 *      verbose = 1
 *
 *
 *  Creating a Preconditioner / Solver
 *  ==================================
 *
 *  There are three ways to create preconditioners and solvers ranging from fine-grained control to fully
 *  automatic creation.
 *
 *  The last option is the most easy to use and should suit most cases.
 *
 *  Explicit Constructor Call
 *  -------------------------
 *  All supported preconditioners and solvers offer constructors that take a specific ParameterTree subsection to read their parameters from.
 *
 *  Use this if you want to call the specific constructor yourself and only want configuration parameters to be
 *  read from the ParameterTree.
 *
 *  Example:
 *
 *      // Create preconditioner
 *      typedef Dune::SeqSSOR<MatrixType,VectorType> Preconditioner;
 *      Preconditioner precond(matrix,configuration.sub("SeqSSOR"));
 *
 *  Specific Factory
 *  ----------------
 *  Both PreconditionerFactory and SolverFactory take a string specifying the type and a ParameterTree subsection
 *  containing the respective configuration values.
 *
 *  Use this if you want to create preconditioners and solvers separately.
 *
 *  Example:
 *
 *      // Create preconditioner
 *      auto prec = Dune::PreconditionerFactory::create<VectorType,VectorType> ("SeqSSOR", matrix, configuration.sub("SeqSSOR"));
 *      // Create solver
 *      auto solver = Dune::SolverFactory::create<VectorType> (linearoperator, prec, "CGSolver", configuration.sub("CGSolver"));
 *
 *
 *  Fully Automatic
 *  ---------------
 *  The SolverPrecondFactory takes a ParameterTree that specifies the solver and preconditioner type as well as
 *  their configuration.
 *
 *  Use this if you want a ready-to-use solver and the types to be configurable in the ParemeterTree.
 *
 *  Example:
 *
 *      // Create solver including preconditioner
 *      auto solver = SolverPrecondFactory::create<VectorType> (matrix, configuration, "solver");
 *
 *  \note Parallel versions of the factory calls exist. They additionally require a communicator to be passed.
 *
 *
 *  ISTL Library - Speed Up Compile Time
 *  ==================================
 *
 *  The ISTL Library mechanism allows building a separate library with preinstantiated preconditioners and solvers
 *  to use with your own module. This can speed up compile times significantly especially when using the ISTL factory.
 *
 *  In order to activate this, you have to add the following line to your module's CMake code, specifying the
 *  block sizes of vectors that you pass to the preconditioners and solvers.
 *
 *      dune_add_istl_library (BLOCKSIZES 2 10)
 *
 *
 *  Additionally, the library must be registered with the dune_enable_all_packages call of your module. The name of
 *  the library is the name of your module with the -istl suffix, for example:
 *
 *      dune_enable_all_packages(MODULE_LIBRARIES my-module-name-istl)
 *
 */

namespace Dune {

  /*!
     \ingroup ISTL_Factory
     \brief Creates preconditioners from ParameterTrees.
   */
  class PreconditionerFactory {
  public:

    /*!
       \brief Creates a sequential preconditioner from a ParameterTree.
       \param id A string matching the type name of the desired preconditioner.
       \param A The matrix to be preconditioned.
       \param configuration A ParameterTree subsection containing configuration parameters for the preconditioner.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X, typename Y, typename M>
    static std::shared_ptr<Preconditioner<X,Y> > create (std::string id, const M& A, const ParameterTree& configuration) {
      if (id == "AMG") {

        std::string smoother = configuration.get<std::string> ("smoother");
        if (smoother == "SeqSSOR") {

          typedef SeqSSOR<M,X,Y> Smoother;

          // AMG requires an operator instead of a matrix, need an adapter here
          typedef MatrixOperator<M,X,Y> MADAPT;
          auto matrixoperator = std::make_shared<MADAPT>(A);

          return std::make_shared<Amg::AMG<MADAPT,X,Smoother> > (matrixoperator, configuration);

        } else if (smoother == "SeqJac") {

          typedef SeqJac<M,X,Y> Smoother;

          // AMG requires an operator instead of a matrix, need an adapter here
          typedef MatrixOperator<M,X,Y> MADAPT;
          auto matrixoperator = std::make_shared<MADAPT>(A);

          return std::make_shared<Amg::AMG<MADAPT,X,Smoother> > (matrixoperator, configuration);

        } else {
          DUNE_THROW(ISTLError, "Factory does not know AMG smoother type '" + smoother + "'!\n");
        }
      }
      if (id == "Richardson")
        return std::make_shared<Richardson<X,Y> > (configuration);
      if (id == "SeqGS")
        return std::make_shared<SeqGS<M,X,Y> > (A, configuration);
      if (id == "SeqILU0")
        return std::make_shared<SeqILU0<M,X,Y> > (A, configuration);
      if (id == "SeqILUn")
        return std::make_shared<SeqILUn<M,X,Y> > (A, configuration);
      if (id == "SeqJac")
        return std::make_shared<SeqJac<M,X,Y> > (A, configuration);
      if (id == "SeqSSOR")
        return std::make_shared<SeqSSOR<M,X,Y> > (A, configuration);
      if (id == "SeqSOR")
        return std::make_shared<SeqSOR<M,X,Y> > (A, configuration);
      DUNE_THROW(ISTLError, "Factory does not know sequential preconditioner type '" + id + "'!\n");
    }

    /*!
       \brief Creates a parallel preconditioner from a ParameterTree.
       \param id A string matching the type name of the desired preconditioner.
       \param A The matrix to be preconditioned.
       \param configuration A ParameterTree subsection containing configuration parameters for the preconditioner.
       \param comm The MPI communicator to be used by the preconditioner.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename COMM, typename M, typename X, typename Y>
    static std::shared_ptr<Preconditioner<X,Y> > create (std::string id, const M& A,
                                                         const ParameterTree& configuration,
                                                         const COMM& comm) {
      // Pass dummy operator to actual implementation
      std::shared_ptr<LinearOperator<X,X> > dummy_linearoperator;
      return create <COMM, M, X, Y> (id, A, configuration, comm, dummy_linearoperator);
    }

    /*!
       \brief Creates a parallel preconditioner from a ParameterTree.
       \param id A string matching the type name of the desired preconditioner.
       \param A The matrix to be preconditioned.
       \param configuration A ParameterTree subsection containing configuration parameters for the preconditioner.
       \param comm The MPI communicator to be used by the preconditioner.
       \param out_linearoperator Returns the linear operator the preconditioner will use.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename COMM, typename M, typename X, typename Y>
    static std::shared_ptr<Preconditioner<X,Y> > create (std::string id, const M& A,
                                                         const ParameterTree& configuration,
                                                         const COMM& comm,
                                                         std::shared_ptr<LinearOperator<X,X> >& out_linearoperator) {
      if (id == "AMG") {

        std::string smoother = configuration.get<std::string> ("smoother");
        if (smoother == "SeqSSOR") {

          typedef Dune::SeqSSOR<M,X,X> Smoother;
          typedef Dune::BlockPreconditioner<X,X,COMM,Smoother> ParSmoother;
          typedef Dune::OverlappingSchwarzOperator<M,X,X,COMM> Operator;

          auto linearoperator = std::make_shared<Operator>(A, comm);
          out_linearoperator = linearoperator;

          return std::make_shared<Amg::AMG<Operator,X,ParSmoother,COMM> > (linearoperator, configuration, comm);
        } else if (smoother == "SeqJac") {

          typedef Dune::SeqJac<M,X,X> Smoother;
          typedef Dune::BlockPreconditioner<X,X,COMM,Smoother> ParSmoother;
          typedef Dune::OverlappingSchwarzOperator<M,X,X,COMM> Operator;

          auto linearoperator = std::make_shared<Operator>(A, comm);
          out_linearoperator = linearoperator;

          return std::make_shared<Amg::AMG<Operator,X,ParSmoother,COMM> > (linearoperator, configuration, comm);
        } else {
          DUNE_THROW(ISTLError, "Factory does not know AMG smoother type '" + smoother + "'!\n");
        }
      }
      DUNE_THROW(ISTLError, "Factory does not know parallel preconditioner type '" + id + "'!\n");
    }

  };

  /*!
     \ingroup ISTL_Factory
     \brief Creates solvers from ParameterTrees, requires preconditioners.
   */
  class SolverFactory {
  public:

    /*!
       \brief Creates a solver from a ParameterTree, requires a preconditioner.
       \param linearoperator The linear operator that the solver will work on.
       \param preconditioner The preconditioner to be given to the solver.
       \param id A string matching the type name of the desired solver.
       \param configuration A ParameterTree subsection containing configuration parameters for the solver.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X>
    static std::shared_ptr<InverseOperator<X,X> > create (std::shared_ptr<LinearOperator<X,X> > linearoperator,
                                                          std::shared_ptr<Preconditioner<X,X> > preconditioner,
                                                          std::string id, const ParameterTree& configuration)
    {
      if (id == "BiCGSTABSolver")
        return std::make_shared<BiCGSTABSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "CGSolver")
        return std::make_shared<CGSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "GeneralizedPCGSolver")
        return std::make_shared<GeneralizedPCGSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "GradientSolver")
        return std::make_shared<GradientSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "LoopSolver")
        return std::make_shared<LoopSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "MINRESSolver")
        return std::make_shared<MINRESSolver<X> > (linearoperator, preconditioner, configuration);
      if (id == "RestartedGMResSolver")
        return std::make_shared<RestartedGMResSolver<X> > (linearoperator, preconditioner, configuration);
      DUNE_THROW(ISTLError, "Factory does not know sequential solver type '" + id + "'!\n");
    }

    /*!
       \brief Creates a solver from a ParameterTree, requires a preconditioner.
       \param linearoperator The linear operator that the solver will work on.
       \param preconditioner The preconditioner to be given to the solver.
       \param id A string matching the type name of the desired solver.
       \param configuration A ParameterTree subsection containing configuration parameters for the solver.
       \param comm The MPI communicator to be used.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X, typename COMM>
    static std::shared_ptr<InverseOperator<X,X> > create (std::shared_ptr<LinearOperator<X,X> > linearoperator,
                                                          std::shared_ptr<Preconditioner<X,X> > preconditioner,
                                                          std::string id, const ParameterTree& configuration,
                                                          const COMM& comm)
    {
      // Create suitable scalar product
      auto scalarproduct = ScalarProductChooser::construct<X> (preconditioner->category(), comm);

      if (id == "BiCGSTABSolver")
        return std::make_shared<BiCGSTABSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "CGSolver")
        return std::make_shared<CGSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "GeneralizedPCGSolver")
        return std::make_shared<GeneralizedPCGSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "GradientSolver")
        return std::make_shared<GradientSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "LoopSolver")
        return std::make_shared<LoopSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "MINRESSolver")
        return std::make_shared<MINRESSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      if (id == "RestartedGMResSolver")
        return std::make_shared<RestartedGMResSolver<X> > (linearoperator, scalarproduct, preconditioner, configuration);
      DUNE_THROW(ISTLError, "Factory does not know parallel solver type '" + id + "'!\n");
    }

  };

  /*!
     \ingroup ISTL_Factory
     \brief Creates ready-to-use solvers with preconditioners from ParameterTrees.
   */
  class SolverPrecondFactory {
  public:

    /*!
       \brief Creates a sequential solver with a preconditioner from a ParameterTree.
       \param A The matrix to be solved.
       \param configuration A ParameterTree containing preconditioner and solver types as well as their configuration.
       \param group The name of the ParameterTree section defining the solver and preconditioner type.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X, typename M>
    static std::shared_ptr<InverseOperator<X,X> > create (const M& A, ParameterTree& configuration,
                                                          std::string group = "solver")
    {

      // Read selection of precond and solver
      auto subconf = configuration.sub (group);
      std::string solver_id = subconf["solver"];
      std::string precond_id = subconf["precond"];

      // Get subtrees of precond and solver
      auto precondconf = configuration.sub (precond_id);
      auto solverconf = configuration.sub (solver_id);

      // If type is specified explicitly, use it
      precond_id = precondconf.get ("type", precond_id);
      solver_id = solverconf.get ("type", solver_id);

      // Adapter needed for solvers, pass as shared pointer to ensure sufficient lifetime
      std::shared_ptr<LinearOperator<X,X> > madapt (new MatrixOperator<M,X,X>(A));

      // Build precond and solver, return solver
      auto preconditioner = PreconditionerFactory::create<X, X> (precond_id, A, precondconf);
      return SolverFactory::create (madapt, preconditioner, solver_id, solverconf);
    }

    /*!
       \brief Creates a parallel solver with a preconditioner from a ParameterTree.
       \param A The matrix to be solved.
       \param comm The communicator to be used.
       \param configuration A ParameterTree containing preconditioner and solver types as well as their configuration.
       \param group The name of the ParameterTree section defining the solver and preconditioner type.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X, typename COMM, typename M>
    static std::shared_ptr<InverseOperator<X,X> > create (const M& A,
                                                          const COMM& comm,
                                                          ParameterTree& configuration,
                                                          std::string group = "solver")
    {

      // Read selection of precond and solver
      auto subconf = configuration.sub (group);

      // Pass to sequential case if parallel is not activated
      if (subconf.get<bool>("parallel", false) == false)
        return create<X>(A, configuration, group);

      std::string solver_id = subconf["solver"];
      std::string precond_id = subconf["precond"];

      // Get subtrees of precond and solver
      auto precondconf = configuration.sub (precond_id);
      auto solverconf = configuration.sub (solver_id);

      // Set verbosity to 0 for all but rank 0
      if (comm.communicator().rank() != 0) {
        precondconf["verbose"] = "0";
        solverconf["verbose"] = "0";
      }

      // If type is specified explicitly, use it
      precond_id = precondconf.get ("type", precond_id);
      solver_id = solverconf.get ("type", solver_id);

      // Need to get the linearoperator from the preconditioner factory in order to pass it to the solver
      std::shared_ptr<LinearOperator<X,X> > linearoperator;

      // Build precond and solver, return solver
      auto preconditioner = PreconditionerFactory::create<COMM, M, X, X> (precond_id, A, precondconf, comm, linearoperator);
      return SolverFactory::create<X> (linearoperator, preconditioner, solver_id, solverconf, comm);
    }

  };

} // end namespace

#endif
