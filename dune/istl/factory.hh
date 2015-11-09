// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FACTORY_HH
#define DUNE_ISTL_FACTORY_HH

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioners.hh>
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
 *      maxit = 5000
 *      verbose = 3
 *
 *  \note The preconditioners / solvers are identified by their class names.
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
 *      precond = MyPrecond42
 *      solver = CGSolver
 *
 *      [MyPrecond42]
 *      type = SeqSSOR
 *      iterations = 1
 *      relaxation = 1.8
 *
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
 *  All preconditioners and solvers offer constructors that take a specific ParameterTree subsection to read their parameters from.
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
 *      // Create solver including a preconditioner
 *      auto prec = Dune::PreconditionerFactory::create<VectorType,VectorType> ("SeqSSOR", matrix, configuration.sub("SeqSSOR"));
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
 *      // Create preconditioner
 *      auto solver = SolverPrecondFactory::create<VectorType> (matrix, configuration, "solver");
 *
 *  \note Here, not only a specific subsection but the entire ParameterTree is required!
 *
 *
 */

namespace Dune {

  /*!
     \ingroup ISTL_Factory
     \brief Creates preconditioners from ParameterTrees
   */
  class PreconditionerFactory {
  public:

    /*!
       \brief Creates a preconditioner from a ParameterTree.
       \param id A string matching the type name of the desired preconditioner.
       \param A The matrix to be preconditioned.
       \param configuration A ParameterTree subsection containing configuration parameters for the preconditioner.

       See \ref ISTL_Factory for the ParameterTree layout and examples.
     */
    template <typename X, typename Y, typename M>
    static std::shared_ptr<Preconditioner<X,Y> > create (std::string id, const M& A, const ParameterTree& configuration) {
      if (id == "AMG") {
        typedef SeqSSOR<M,X,Y> Smoother;

        // AMG requires an operator instead of a matrix, need an adapter here
        typedef MatrixAdapter<M,X,Y> MADAPT;
        auto matrixadapter = std::make_shared<MADAPT>(A);

        return std::make_shared<Amg::AMG<MADAPT,X,Smoother> > (matrixadapter, configuration);
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
      DUNE_THROW(ISTLError, "Factory does not know preconditioner type '" + id + "'!\n");
    }
  };

  /*!
     \ingroup ISTL_Factory
     \brief Creates solvers from ParameterTrees, requires preconditioners
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
      DUNE_THROW(ISTLError, "Factory does not know solver type '" + id + "'!\n");
    }
  };

  /*!
     \ingroup ISTL_Factory
     \brief Creates ready-to-use solvers with preconditioners from ParameterTrees
   */
  class SolverPrecondFactory {
  public:

    /*!
       \brief Creates a solver with a preconditioner from a ParameterTree.
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
      std::shared_ptr<LinearOperator<X,X> > madapt (new MatrixAdapter<M,X,X>(A));

      // Build precond and solver, return solver
      auto preconditioner = PreconditionerFactory::create<X, X> (precond_id, A, precondconf);
      return SolverFactory::create (madapt, preconditioner, solver_id, solverconf);
    }
  };

} // end namespace

#endif
