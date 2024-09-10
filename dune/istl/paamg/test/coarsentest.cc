// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/paamg/matrixhierarchy.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#include "anisotropic.hh"

#include <utility>

constexpr int blockSize = 10;

using Communication = Dune::OwnerOverlapCopyCommunication<int,int>;
using ParallelIndexSet = Communication::ParallelIndexSet;
using MatrixBlock = Dune::FieldMatrix<double,blockSize,blockSize>;
using BCRSMat = Dune::BCRSMatrix<MatrixBlock>;
using VectorBlock = Dune::FieldVector<double,blockSize>;
using Vector = Dune::BlockVector<VectorBlock>;

/**
 * Check if the matrices A and B are the same.
 *
 * @param BCRSMat A
 * @param BCRSMat B
 * @return bool - True if A and B are the same, false if they are not.
 */
bool areSame(BCRSMat A, BCRSMat B)
{
  for (auto rowIt = A.begin(); rowIt != A.end(); ++rowIt)
    for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
      for (size_t k = 0; k < blockSize; k++)
        for (size_t l = 0; l < blockSize; l++) {
          std::ignore /*entryA*/ = (*colIt)[k][l];
          try {
            std::ignore /*entryB*/ = B[rowIt.index()][colIt.index()][k][l];
          } catch (Dune::ISTLError& e) {
            return false; // If the entry B[rowIt.index()][colIt.index()][k][l] does not exist, then the matrices are not the same.
          }
          if (std::abs((*colIt)[k][l]) > 1e-7 and std::abs(B[rowIt.index()][colIt.index()][k][l]) > 1e-7
              and std::abs((*colIt)[k][l] - B[rowIt.index()][colIt.index()][k][l])/(*colIt)[k][l] > 1e-7) {
            return false; // Check if the entries are approximately the same.
          }
        }
  return true;
}

/**
 * Builds two AMG matrix hierarchies and checks if they are equal.
 * When setting up such matrix hierarchies on multiple processes, they get created in a non-deterministic way and the two matrix hierarchies might differ.
 *
 * @param int N Total number of unknowns
 * @param int rank Rank of current process
 * @param int noProcesses Number of all processes
 * @return int - 0 if the hierarchies are the same, 1 if they are not.
 */
int testHierarchy(int N, int rank, int noProcesses)
{
  int n;
  Communication pinfo(Dune::MPIHelper::getCommunicator());
  ParallelIndexSet& indices = pinfo.indexSet();

  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;
  RemoteIndices& remoteIndices = pinfo.remoteIndices();

  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, indices, Dune::MPIHelper::getCommunication(), &n);

  remoteIndices.rebuild<false>();

  using OverlapFlags = Dune::NegateSet<Communication::OwnerSet>;
  using Operator = Dune::OverlappingSchwarzOperator<BCRSMat,Vector,Vector,Communication>;
  using Hierarchy = Dune::Amg::MatrixHierarchy<Operator,Communication>;

  Operator op(mat, pinfo);
  Hierarchy hierarchyA(
    Dune::stackobject_to_shared_ptr(op),
    Dune::stackobject_to_shared_ptr(pinfo));

  Hierarchy hierarchyB(
    Dune::stackobject_to_shared_ptr(op),
    Dune::stackobject_to_shared_ptr(pinfo));

  using Criterion = Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >;

  Criterion criterionA(1000,10,1.2,1.6,Dune::Amg::AccumulationMode::successiveAccu,true);
  Criterion criterionB(1000,10,1.2,1.6,Dune::Amg::AccumulationMode::successiveAccu,true);

  hierarchyA.template build<OverlapFlags>(criterionA);
  hierarchyB.template build<OverlapFlags>(criterionB);

  MPI_Barrier(MPI_COMM_WORLD);
  //Check if the second finest matrices are the same
  auto finestMatrixA = hierarchyA.matrices().finest();
  finestMatrixA++;
  auto finestMatrixB = hierarchyB.matrices().finest();
  finestMatrixB++;
  if (!areSame(finestMatrixA->getmat(), finestMatrixB->getmat())) {
      std::cerr << "During run with " << noProcesses << " processes: The matrix hierarchy on rank " << rank << " was created in a non-deterministic way." << std::endl;
      return 1;
  }
  return 0;
}


int main(int argc, char** argv)
{
  Dune::MPIHelper& mpiHelper = Dune::MPIHelper::instance(argc, argv);
  int N=10;

  if(argc>1)
    N = atoi(argv[1]);

  return testHierarchy(N, mpiHelper.rank(), mpiHelper.size());
}
