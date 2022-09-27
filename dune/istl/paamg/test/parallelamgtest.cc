// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#ifdef TEST_AGGLO
#define UNKNOWNS 10
#endif
#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/mpicommunication.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <string>

template<class T, class C>
class DoubleStepPreconditioner
  : public Dune::Preconditioner<typename T::domain_type, typename T::range_type>
{
public:
  typedef typename T::domain_type X;
  typedef typename T::range_type Y;

  enum {category = T::category};

  DoubleStepPreconditioner(T& preconditioner_, C& comm)
    : preconditioner(&preconditioner_), comm_(comm)
  {}

  virtual void pre (X& x, Y& b)
  {
    preconditioner->pre(x,b);
  }

  virtual void apply(X& v, const Y& d)
  {
    preconditioner->apply(v,d);
    comm_.copyOwnerToAll(v,v);
  }

  virtual void post (X& x)
  {
    preconditioner->post(x);
  }
private:
  T* preconditioner;
  C& comm_;
};


class MPIError {
public:
  /** @brief Constructor. */
  MPIError(std::string s, int e) : errorstring(s), errorcode(e){}
  /** @brief The error string. */
  std::string errorstring;
  /** @brief The mpi error code. */
  int errorcode;
};

void MPI_err_handler([[maybe_unused]] MPI_Comm *comm, int *err_code, ...)
{
  char *err_string=new char[MPI_MAX_ERROR_STRING];
  int err_length;
  MPI_Error_string(*err_code, err_string, &err_length);
  std::string s(err_string, err_length);
  std::cerr << "An MPI Error occurred:"<<std::endl<<s<<std::endl;
  delete[] err_string;
  throw MPIError(s, *err_code);
}

template<int BS>
void testAmg(int N, int coarsenTarget)
{
  std::cout<<"==================================================="<<std::endl;
  std::cout<<"BS="<<BS<<" N="<<N<<" coarsenTarget="<<coarsenTarget<<std::endl;

  int procs, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef int GlobalId;
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;
  typedef Dune::OverlappingSchwarzOperator<BCRSMat,Vector,Vector,Communication> Operator;
  int n;

  N/=BS;

  Communication comm(MPI_COMM_WORLD);

  BCRSMat mat = setupAnisotropic2d<MatrixBlock>(N, comm.indexSet(), comm.communicator(), &n, 1);

  const BCRSMat& cmat = mat;

  comm.remoteIndices().template rebuild<false>();

  Vector b(cmat.N()), x(cmat.M());

  b=0;
  x=100;

  setBoundary(x, b, N, comm.indexSet());

  Vector b1=b, x1=x;

  if(N<=6) {
    std::ostringstream name;
    name<<rank<<": row";

    Dune::printmatrix(std::cout, cmat, "A", name.str().c_str());
    Dune::printvector(std::cout, x, "x", name.str().c_str());
    //Dune::printvector(std::cout, b, "b", name.str().c_str());
    //Dune::printvector(std::cout, b1, "b1", "row");
    //Dune::printvector(std::cout, x1, "x1", "row");
  }

  Dune::Timer watch;

  watch.reset();
  Operator fop(cmat, comm);

  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<BCRSMat,Dune::Amg::FirstDiagonal> >
  Criterion;
  typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqJac<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqILU0<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqILUn<BCRSMat,Vector,Vector> Smoother;
  typedef Dune::BlockPreconditioner<Vector,Vector,Communication,Smoother> ParSmoother;
  typedef typename Dune::Amg::SmootherTraits<ParSmoother>::Arguments SmootherArgs;

  Dune::OverlappingSchwarzScalarProduct<Vector,Communication> sp(comm);

  Dune::InverseOperatorResult r, r1;

  double buildtime;

  SmootherArgs smootherArgs;

  smootherArgs.iterations = 1;


  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);


  typedef Dune::Amg::AMG<Operator,Vector,ParSmoother,Communication> AMG;

  AMG amg(fop, criterion, smootherArgs, comm);

  buildtime = watch.elapsed();

  if(rank==0)
    std::cout<<"Building hierarchy took "<<buildtime<<" seconds"<<std::endl;

  Dune::CGSolver<Vector> amgCG(fop, sp, amg, 10e-8, 300, (rank==0) ? 2 : 0);
  watch.reset();

  amgCG.apply(x,b,r);
  amg.recalculateHierarchy();


  MPI_Barrier(MPI_COMM_WORLD);
  double solvetime = watch.elapsed();

  b=0;
  x=100;

  setBoundary(x, b, N, comm.indexSet());

  Dune::CGSolver<Vector> amgCG1(fop, sp, amg, 10e-8, 300, (rank==0) ? 2 : 0);
  amgCG1.apply(x,b,r);

  if(!r.converged && rank==0)
    std::cerr<<" AMG Cg solver did not converge!"<<std::endl;

  if(rank==0) {
    std::cout<<"AMG solving took "<<solvetime<<" seconds"<<std::endl;

    std::cout<<"AMG building took "<<(buildtime/r.elapsed*r.iterations)<<" iterations"<<std::endl;
    std::cout<<"AMG building together with slving took "<<buildtime+solvetime<<std::endl;
  }

}

template<int BSStart, int BSEnd, int BSStep=1>
struct AMGTester
{
  static void test(int N, int coarsenTarget)
  {
    testAmg<BSStart>(N, coarsenTarget);
    const int next = (BSStart+BSStep>BSEnd) ? BSEnd : BSStart+BSStep;
    AMGTester<next,BSEnd,BSStep>::test(N, coarsenTarget);
  }
}
;

template<int BSStart,int BSStep>
struct AMGTester<BSStart,BSStart,BSStep>
{
  static void test(int N, int coarsenTarget)
  {
    testAmg<BSStart>(N, coarsenTarget);
  }
};


int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Errhandler handler;
  MPI_Comm_create_errhandler(MPI_err_handler, &handler);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);

  int N=100;

  int coarsenTarget=200;

  if(argc>1)
    N = atoi(argv[1]);

  if(argc>2)
    coarsenTarget = atoi(argv[2]);

#ifdef TEST_AGGLO
  N=UNKNOWNS;
#endif
  AMGTester<1,1>::test(N, coarsenTarget);
  //AMGTester<1,5>::test(N, coarsenTarget);
  //  AMGTester<10,10>::test(N, coarsenTarget);

  MPI_Finalize();
}
