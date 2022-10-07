// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

/*#include"xreal.h"

   namespace std
   {

   HPA::xreal abs(const HPA::xreal& t)
   {
    return t>=0?t:-t;
   }

   };
 */

#include "anisotropic.hh"
#include <dune/common/timer.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>
#include <dune/istl/solvers.hh>
#include <cstdlib>
#include <ctime>
#include <complex>

typedef double XREAL;
// typedef std::complex<double> XREAL;

/*
   typedef HPA::xreal XREAL;

   namespace Dune
   {
   template<>
   struct DoubleConverter<HPA::xreal>
   {
   static double toDouble(const HPA::xreal& t)
   {
     return t._2double();
   }
   };
   }
 */

namespace Dune
{
  using Mat = BCRSMatrix<FieldMatrix<double,1,1>>;
  using Vec = BlockVector<FieldVector<double,1>>;
  using LinOp = MatrixAdapter<Mat,Vec,Vec>;
  using Comm = Amg::SequentialInformation;

  // explicit template instantiation of FastAMG preconditioner
  template class Amg::AMG<LinOp, Vec, Richardson<Vec,Vec>, Comm>;
  template class Amg::AMG<LinOp, Vec, SeqJac<Mat,Vec,Vec>, Comm>;
  template class Amg::AMG<LinOp, Vec, SeqSOR<Mat,Vec,Vec>, Comm>;
  template class Amg::AMG<LinOp, Vec, SeqSSOR<Mat,Vec,Vec>, Comm>;

} // end namespace Dune


template<class M, class V>
void randomize(const M& mat, V& b)
{
  V x=b;

  srand((unsigned)std::clock());

  typedef typename V::iterator iterator;
  for(iterator i=x.begin(); i != x.end(); ++i)
    *i=(rand() / (RAND_MAX + 1.0));

  mat.mv(static_cast<const V&>(x), b);
}


template <class Matrix, class Vector>
Dune::InverseOperatorResult testAMG(int N, int coarsenTarget, int ml, int gamma = 1)
{

  std::cout<<"N="<<N<<" coarsenTarget="<<coarsenTarget<<" maxlevel="<<ml<<std::endl;


  typedef Dune::ParallelIndexSet<int,LocalIndex,512> ParallelIndexSet;

  ParallelIndexSet indices;
  typedef Dune::MatrixAdapter<Matrix,Vector,Vector> Operator;
  typedef Dune::Communication<void*> Comm;
  int n;

  Comm c;
  Matrix mat = setupAnisotropic2d<typename Matrix::block_type>(N, indices, c, &n, 1);

  Vector b(mat.N()), x(mat.M());

  b=0;
  x=100;

  setBoundary(x, b, N);

  x=0;
  randomize(mat, b);

  if(N<6) {
    Dune::printmatrix(std::cout, mat, "A", "row");
    Dune::printvector(std::cout, x, "x", "row");
  }

  Dune::Timer watch;

  watch.reset();
  Operator fop(mat);

  typedef typename std::conditional< std::is_convertible<XREAL, typename Dune::FieldTraits<XREAL>::real_type>::value,
                   Dune::Amg::FirstDiagonal, Dune::Amg::RowSum >::type Norm;
  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Matrix,Norm> >
          Criterion;
  typedef Dune::SeqSSOR<Matrix,Vector,Vector> Smoother;
  //typedef Dune::SeqSOR<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqJac<BCRSMat,Vector,Vector> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::MultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector,Dune::SymmetricMultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,Vector> Smoother;
  typedef typename Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;

  SmootherArgs smootherArgs;

  smootherArgs.iterations = 1;

  //smootherArgs.overlap=SmootherArgs::vertex;
  //smootherArgs.overlap=SmootherArgs::none;
  //smootherArgs.overlap=SmootherArgs::aggregate;

  smootherArgs.relaxationFactor = 1;

  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setGamma(gamma);
  criterion.setMaxLevel(ml);
  criterion.setSkipIsolated(false);
  // specify pre/post smoother steps
  criterion.setNoPreSmoothSteps(1);
  criterion.setNoPostSmoothSteps(1);

  Dune::SeqScalarProduct<Vector> sp;
  typedef Dune::Amg::AMG<Operator,Vector,Smoother> AMG;

  Smoother smoother(mat,1,1);

  AMG amg(fop, criterion, smootherArgs);


  double buildtime = watch.elapsed();

  std::cout<<"Building hierarchy took "<<buildtime<<" seconds"<<std::endl;

  Dune::GeneralizedPCGSolver<Vector> amgCG(fop,amg,1e-6,80,2);
  //Dune::LoopSolver<Vector> amgCG(fop, amg, 1e-4, 10000, 2);
  watch.reset();
  Dune::InverseOperatorResult r;
  amgCG.apply(x,b,r);

  XREAL solvetime = watch.elapsed();

  std::cout<<"AMG solving took "<<solvetime<<" seconds"<<std::endl;

  std::cout<<"AMG building took the same time as "<<(buildtime/r.elapsed*r.iterations)<<" iterations"<<std::endl;
  std::cout<<"AMG building together with solving took "<<buildtime+solvetime<<std::endl;

  /*
     watch.reset();
     cg.apply(x,b,r);

     std::cout<<"CG solving took "<<watch.elapsed()<<" seconds"<<std::endl;
   */
  return r;
}


int main(int argc, char** argv)
try
{
  int N=100;
  int coarsenTarget=1200;
  int ml=10;

  if(argc>1)
    N = atoi(argv[1]);

  if(argc>2)
    coarsenTarget = atoi(argv[2]);

  if(argc>3)
    ml = atoi(argv[3]);

  Dune::InverseOperatorResult gamma1_res;
  for(int gamma = 1; gamma<=2;++gamma){
    {
      using Matrix = Dune::BCRSMatrix<XREAL>;
      using Vector = Dune::BlockVector<XREAL>;

      Dune::InverseOperatorResult res = testAMG<Matrix,Vector>(N, coarsenTarget, ml, gamma);
      if(gamma==1){
        gamma1_res = res;
      }else{
        assert(res.conv_rate < gamma1_res.conv_rate);
      }
    }
  }

  {
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<XREAL,1,1> >;
    using Vector = Dune::BlockVector<Dune::FieldVector<XREAL,1> >;

    testAMG<Matrix,Vector>(N, coarsenTarget, ml);
  }

  {
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<XREAL,2,2> >;
    using Vector = Dune::BlockVector<Dune::FieldVector<XREAL,2> >;

    testAMG<Matrix,Vector>(N, coarsenTarget, ml);
  }

  return 0;
}
catch (std::exception &e)
{
  std::cout << "ERROR: " << e.what() << std::endl;
  return 1;
}
catch (...)
{
  std::cerr << "Dune reported an unknown error." << std::endl;
  exit(1);
}
