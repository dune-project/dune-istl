// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include "factorytest_anisotropic.hh"
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/factory.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <string>

bool test(std::string test_ini_id)
{
  // Choose vector/matrix types
  const int BS = 10;

  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;


  // Setup communication
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;

  Communication comm(MPI_COMM_WORLD);


  // Setup test problem
  int N = 100;
  N/=BS;
  int n;
  BCRSMat mat = setupAnisotropic2d<BCRSMat>(N, comm.indexSet(), comm.communicator(), &n, 1);
  BCRSMat& cmat = mat;

  comm.remoteIndices().template rebuild<false>();

  Vector b(cmat.N()), x(cmat.M());

  b=0;
  x=100;

  setBoundary(x, b, N, comm.indexSet());


  // Load configuration
  Dune::ParameterTree configuration;
  Dune::ParameterTreeParser parser;
  parser.readINITree("factorytest_config.ini", configuration);


  // Get solver from factory
  auto solver = Dune::SolverPrecondFactory::create<Vector> (cmat, comm, configuration, test_ini_id);


  // Solve
  Dune::InverseOperatorResult r;
  solver->apply(x,b,r);


  // Check if converged
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(!r.converged && rank == 0) {
    std::cerr << " Solver did not converge!" << std::endl;
    return false;
  }
  return true;
}


int main(int argc, char** argv)
{
  try {
    MPI_Init(&argc, &argv);

    // Test different configurations (all defined in the ini file)
    if (!test("test1")) return -1;
    if (!test("test2")) return -1;
    if (!test("test3")) return -1;

    MPI_Finalize();
    return 0;
  }
  catch (Dune::Exception &e)
  {
    std::cerr << "Dune reported error: " << e << std::endl;
    return -1;
  }
  catch (...)
  {
    return -1;
  }
}
