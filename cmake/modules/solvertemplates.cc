#include <config.h>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/owneroverlapcopy.hh> // For instantiation of parallel factories
#include <dune/istl/factory.hh>

namespace Dune {

#if HAVE_MPI

  namespace Precomp${BLOCKSIZE} {

    typedef Dune::BlockVector<Dune::FieldVector<double,${BLOCKSIZE}> > V;
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,${BLOCKSIZE},${BLOCKSIZE}> > M;
    typedef OwnerOverlapCopyCommunication<int> COMM;

    typedef Dune::OverlappingSchwarzOperator<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::COMM> Operator1;
    typedef Dune::SeqSSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> Smoother1;
    typedef Dune::BlockPreconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::COMM,Smoother1> ParSmoother1;

  }
  template class Amg::AMG<Precomp${BLOCKSIZE}::Operator1,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::ParSmoother1,Precomp${BLOCKSIZE}::COMM>;

  template std::shared_ptr<InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverPrecondFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::COMM, Precomp${BLOCKSIZE}::M>(const Precomp${BLOCKSIZE}::M& A, const Precomp${BLOCKSIZE}::COMM& comm, ParameterTree& configuration, std::string group);
  template std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::PreconditionerFactory::create<Precomp${BLOCKSIZE}::COMM, Precomp${BLOCKSIZE}::M, Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::V>(std::string id, const Precomp${BLOCKSIZE}::M& A, const ParameterTree& configuration, const Precomp${BLOCKSIZE}::COMM& comm, std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> >& out_linearoperator);
  template std::shared_ptr<InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverFactory::create<Precomp${BLOCKSIZE}::V>(std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > linearoperator, std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > preconditioner, std::string id, const ParameterTree& configuration, const Precomp${BLOCKSIZE}::COMM& comm);

#endif

  template std::shared_ptr<Dune::InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverPrecondFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::M>(const Precomp${BLOCKSIZE}::M& A, ParameterTree& configuration, std::string group);
  template std::shared_ptr<Dune::Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::PreconditionerFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::M>(std::string id, const Precomp${BLOCKSIZE}::M& A, const ParameterTree& configuration);
  template std::shared_ptr<Dune::InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverFactory::create<Precomp${BLOCKSIZE}::V>(std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > linearoperator, std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > preconditioner, std::string id, const ParameterTree& configuration);


  template class Dune::BiCGSTABSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::CGSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::GeneralizedPCGSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::GradientSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::LoopSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::MINRESSolver<Precomp${BLOCKSIZE}::V>;
  template class Dune::RestartedGMResSolver<Precomp${BLOCKSIZE}::V>;

  template class Amg::AMG<MatrixOperator<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>,Precomp${BLOCKSIZE}::V,SeqSSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> >;
  template class Dune::Richardson<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqGS<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqILU0<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqILUn<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqJac<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqSSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  template class Dune::SeqSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;

}
