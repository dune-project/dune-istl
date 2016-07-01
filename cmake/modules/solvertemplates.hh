#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/owneroverlapcopy.hh> // For instantiation of parallel factories
#include <dune/istl/factory.hh>

#ifndef DUNE_ISTL_SOLVERTEMPLATES${BLOCKSIZE}_HH
#define DUNE_ISTL_SOLVERTEMPLATES${BLOCKSIZE}_HH

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
  extern template class Amg::AMG<Precomp${BLOCKSIZE}::Operator1,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::ParSmoother1,Precomp${BLOCKSIZE}::COMM>;

  extern template std::shared_ptr<InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverPrecondFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::COMM, Precomp${BLOCKSIZE}::M>(const Precomp${BLOCKSIZE}::M& A, const Precomp${BLOCKSIZE}::COMM& comm, ParameterTree& configuration, std::string group);
  extern template std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::PreconditionerFactory::create<Precomp${BLOCKSIZE}::COMM, Precomp${BLOCKSIZE}::M, Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::V>(std::string id, const Precomp${BLOCKSIZE}::M& A, const ParameterTree& configuration, const Precomp${BLOCKSIZE}::COMM& comm, std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> >& out_linearoperator);
  extern template std::shared_ptr<InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverFactory::create<Precomp${BLOCKSIZE}::V>(std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > linearoperator, std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > preconditioner, std::string id, const ParameterTree& configuration, const Precomp${BLOCKSIZE}::COMM& comm);

#endif

  extern template std::shared_ptr<Dune::InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverPrecondFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::M>(const Precomp${BLOCKSIZE}::M& A, ParameterTree& configuration, std::string group);
  extern template std::shared_ptr<Dune::Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::PreconditionerFactory::create<Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::V, Precomp${BLOCKSIZE}::M>(std::string id, const Precomp${BLOCKSIZE}::M& A, const ParameterTree& configuration);
  extern template std::shared_ptr<Dune::InverseOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > Dune::SolverFactory::create<Precomp${BLOCKSIZE}::V>(std::shared_ptr<LinearOperator<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > linearoperator, std::shared_ptr<Preconditioner<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> > preconditioner, std::string id, const ParameterTree& configuration);


  extern template class Dune::BiCGSTABSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::CGSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::GeneralizedPCGSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::GradientSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::LoopSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::MINRESSolver<Precomp${BLOCKSIZE}::V>;
  extern template class Dune::RestartedGMResSolver<Precomp${BLOCKSIZE}::V>;

  extern template class Amg::AMG<MatrixOperator<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>,Precomp${BLOCKSIZE}::V,SeqSSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V> >;
  extern template class Dune::Richardson<Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqGS<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqILU0<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqILUn<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqJac<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqSSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;
  extern template class Dune::SeqSOR<Precomp${BLOCKSIZE}::M,Precomp${BLOCKSIZE}::V,Precomp${BLOCKSIZE}::V>;

}

#endif
