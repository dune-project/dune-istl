// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <config.h>

// Uncomment the following line if you want to use direct solvers, but have MPI installed on you system
//#undef HAVE_MPI

#include <fenv.h>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixmarket.hh>

// include solvers and preconditioners for the solverfactory
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/ldl.hh>
#include <dune/istl/spqr.hh>
#include <dune/istl/paamg/amg.hh>


#include "istl-solver-playground.hh"

using VectorFieldType = double;

typedef Dune::BCRSMatrix<Dune::FieldMatrix<Dune::Simd::Scalar<VectorFieldType>,1,1>> Mat;
typedef Dune::BlockVector<Dune::FieldVector<VectorFieldType,1>> Vec;

void printHelp(){
  std::cout << "istl-playground" << std::endl
            << "Loads and solves a system from MatrixMarket" <<std::endl
            << "format and stores the result in MatrixMarket format." << std::endl
            << "This program is thought as test environment to play" << std::endl
            << "around with different solvers and preconditioners." << std::endl
            << "Parameters are read in a ParameterTree from playground.ini" << std::endl
            << "but can also be passed in by command line arguments." << std::endl
            << std::endl
            << std::setw(20) << std::left
            << "-matrix"
            << "Filename of the MatrixMarket file of the operator matrix" << std::endl
            << std::setw(20) << std::left
            << ""
            << "(default \"laplacian\", generated from test/laplacian.hh)" << std::endl
            << std::setw(20) << std::left
            << "-rhs"
            << "Filename of the MatrixMarket file of the right-hand side" << std::endl
            << std::setw(20) << std::left
            << "-verbose"
            << "Verbosity (default: 1)" << std::endl
            << std::setw(20) << std::left
            << "-random_rhs"
            << "If 1 generates a random RHS instead of reading it from a file (default: 0)" << std::endl
            << std::setw(20) << std::left
            << "-distributed"
            << "Loads a distributed MatrixMarket format (see matrixmarket.hh) (default: 0)" << std::endl
            << std::setw(20) << std::left
            << "-redistribute"
            << "Redistributes the matrix using ParMETIS (default: 0)" << std::endl
            << std::setw(20) << std::left
            << "-ini"
            << "Filename of the ini-file (default:playground.ini)" << std::endl
            << std::setw(20) << std::left
            << "-check_residual"
            << "Whether to compute the defect at the end (default: 1)" << std::endl
            << std::setw(20) << std::left
            << "-output"
            << "Filename of the output filename in which the result is stored" << std::endl
            << std::setw(20) << std::left
            << "-FP_EXCEPT"
            << "Enables floating point exceptions (default: 0)" << std::endl
            << std::endl
            << "The subtree 'solver' in the ParameterTree is passed to the SolverFactory" << std::endl;
}

int main(int argc, char** argv){
  auto& mpihelper = Dune::MPIHelper::instance(argc, argv);

  if(argc > 1 && argv[1][0]=='-' && argv[1][1] == 'h'){
    printHelp();
    return 0;
  }

  Dune::ParameterTreeParser::readOptions(argc, argv, config);
  Dune::ParameterTreeParser::readINITree(config.get("ini","playground.ini"), config, false);

  if(mpihelper.size() > 1 && !config.get("distributed", false) && !config.get("redistribute", false)){
    std::cerr << "To run this program in parallel you either need to load a distributed"
              << " system (-distributed) or redistribute the system using parmetis"
              << " (-redistribute)." << std::endl;
    return 1;
  }

  if(config.get("FP_EXCEPT", false))
     feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);// | FE_UNDERFLOW);

  int verbose = config.get("verbose", 1);
  if(mpihelper.rank() > 0)
    verbose = 0;
  std::shared_ptr<Vec> rhs = std::make_shared<Vec>();
  std::shared_ptr<Mat> m = std::make_shared<Mat>();

  if(verbose){
    std::cout << "Processes: " << mpihelper.size() << std::endl;
    std::cout << "Loading system... " << std::flush;
  }
  Dune::Timer t;
#if HAVE_MPI
  auto oocomm = loadSystem(m, rhs, config, mpihelper.getCommunication());
  typedef Dune::OverlappingSchwarzOperator<Mat, Vec, Vec, OOCOMM> OP;
  typedef Dune::OverlappingSchwarzScalarProduct<Vec, OOCOMM> SP;
  std::shared_ptr<OP> op = std::make_shared<OP>(*m, *oocomm);
  std::shared_ptr<SP> sp = std::make_shared<SP>(*oocomm);
#else
  loadSystem(m,rhs,config);
  typedef Dune::MatrixAdapter<Mat, Vec, Vec> OP;
  typedef Dune::ScalarProduct<Vec> SP;
  std::shared_ptr<OP> op = std::make_shared<OP>(*m);
  std::shared_ptr<SP> sp = std::make_shared<SP>();
#endif
  if(verbose)
    std::cout << t.elapsed() << " s" << std::endl;

  if(config.get("redistribute", false) && mpihelper.size() > 1){
#if HAVE_PARMETIS && HAVE_MPI
    Dune::Timer t;
    if(verbose > 0)
      std::cout << "Redistributing...  " << std::flush;
    redistribute(m, rhs, oocomm);
    op = std::make_shared<OP>(*m, *oocomm);
    sp = std::make_shared<SP>(*oocomm);
    if(verbose > 0)
      std::cout << t.elapsed() << " s" << std::endl;
#else
      std::cerr << "ParMETIS is necessary to redistribute the matrix." << std::endl;
#endif
  }

  if(rhs->size() != m->N()){
    std::cerr << "Dimensions of matrix and rhs do not match." << std::endl;
    return 1;
  }

  Vec x(m->M());
  typedef typename Vec::field_type field_type;
  x=field_type(0.0);

  // one output is enough!
  if(mpihelper.rank()!=0)
    config["solver.verbose"] = "0";

  std::unique_ptr<Vec> pRHS;
  bool check_residual = config.get("check_residual", true);
  if(check_residual)
    pRHS = std::make_unique<Vec>(*rhs);

  solve(op, *rhs, x, config.sub("solver"),verbose);

  if(check_residual){
    op->applyscaleadd(-1.0, x, *pRHS);
    auto defect = sp->norm(*pRHS);
    if(verbose > 0)
      std::cout << "Final defect: " << Dune::Simd::io(defect) << std::endl;
  }

  std::string outputfilename = config.get<std::string>("output", "");
  if(outputfilename.size() != 0){
#if HAVE_MPI
    Dune::storeMatrixMarket(x, outputfilename, *oocomm, true);
#else
    std::ostringstream rfilename;
    rfilename << outputfilename << ".mm";
    Dune::storeMatrixMarket(x, rfilename.str());
#endif
  }
  return 0;
}
