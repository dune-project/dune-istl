// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \author: Matthias Wohlmuth
 * \file
 * \brief Test different solvers and preconditioners for
 *        \f$A*x = b\f$ with \f$A\f$ being a \f$N^2 \times N^2\f$
 *        Laplacian and \f$b\f$ a complex valued rhs.
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include <complex>

#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include "laplacian.hh"
#include "assemble_random_matrix_vectors.hh"

#if HAVE_SUPERLU
#include <dune/istl/superlu.hh>
static_assert(SUPERLU_NTYPE == 3,
  "If SuperLU is selected for complex rhs test, then SUPERLU_NTYPE must be set to 3 (std::complex<double>)!");
#endif

typedef std::complex<double> FIELD_TYPE;

/**
 * \brief Test different solvers and preconditioners for
 *        \f$A*x = b\f$ with \f$A\f$ being a \f$N^2 \times N^2\f$
 *        Laplacian and \f$b\f$ a complex valued rhs.
 *
 * The rhs and reference solutions were computed using the following matlab code:
 * \code
    N=3;
    A = full(gallery('poisson',N)); % create poisson matrix

    % find a solution consiting of complex integers
    indVec = (0:(N*N-1))',
    iVec = complex(0,1).^indVec + indVec,
    x0 = iVec .* indVec,

    % compute corresponding rhs
    b = A * x0,

    % solve system using different solvers
    xcg = pcg(A,b),
    x = A \ b,
    xgmres = gmres(A,b)
 * \endcode
 */
template<class Operator, class Vector>
class SolverTest
{
public:
  SolverTest(Operator & op, Vector & rhs, Vector & x0, double maxError = 1e-10)
  : m_op(op),
    m_x(rhs),
    m_x0(x0),
    m_b(rhs),
    m_rhs(rhs),
    m_maxError(maxError),
    m_numTests(0),
    m_numFailures(0)
  {
    std::cout << "SolverTest uses rhs: " << std::endl
      << m_rhs << std::endl
      << "and expects the solultion: " << m_x0 << std::endl << std::endl;
  }

  template<class Solver>
  bool operator() (Solver & solver)
  {
    m_b = m_rhs;
    m_x = 0;
    solver.apply(m_x, m_b, m_res);
    std::cout << "Defect reduction is " << m_res.reduction << std::endl;
    std::cout << "Computed solution is: " << std::endl;
    std::cout << m_x << std::endl;
    m_b = m_x0;
    m_b -= m_x;
    const double errorNorm = m_b.two_norm();
    std::cout << "Error = " << errorNorm << std::endl;
    ++m_numTests;
    if(errorNorm > m_maxError)
    {
      std::cout << "SolverTest did not converge!" << std::endl;
      ++m_numFailures;
      return false;
    }
    return true;
  }

  int getNumTests() const
  {
    return m_numTests;
  }

  int getNumFailures() const
  {
    return m_numFailures;
  }

private:
  const Operator & m_op;
  Vector m_x, m_x0, m_b;
  const Vector m_rhs;
  double m_maxError;
  int m_numTests, m_numFailures;
  Dune::InverseOperatorResult m_res;
};

int main(int argc, char** argv)
{
  const int BS = 1;
  std::size_t N = 3;

  const int maxIter = int(N*N*N*N);
  const double reduction = 1e-16;

  if (argc > 1)
    N = atoi(argv[1]);
  std::cout<<"testing for N="<<N<<" BS="<<1<<std::endl;

  typedef Dune::FieldMatrix<FIELD_TYPE,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<FIELD_TYPE,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> Vector;
  typedef Dune::MatrixAdapter<BCRSMat,Vector,Vector> Operator;

  BCRSMat mat;
  Operator fop(mat);
  Vector b(N*N), b0(N*N), x0(N*N), x(N*N), error(N*N);

  setupLaplacian(mat,N);

  typedef Vector::Iterator VectorIterator;

  FIELD_TYPE count(0);

  const FIELD_TYPE I(0.,1.); // complex case

  for (VectorIterator it = x0.begin(); it != x0.end(); ++it)
  {
    *it = (count + std::pow(I,count))*count;
    count = count + 1.;
  }
  std::cout << "x0 = " << x0 << std::endl;

  mat.mv(x0,b);

  std::cout << "b = " << b << std::endl;
  b0 = b;
  x = 0;

  Dune::Timer watch;

  watch.reset();

  // create Dummy Preconditioner
  typedef Dune::Richardson<Vector,Vector> DummyPreconditioner;

  typedef Dune::SeqJac<BCRSMat,Vector,Vector> JacobiPreconditioner;
  typedef Dune::SeqGS<BCRSMat,Vector,Vector> GaussSeidelPreconditioner;
  typedef Dune::SeqSOR<BCRSMat,Vector,Vector> SORPreconditioner;
  typedef Dune::SeqSSOR<BCRSMat,Vector,Vector> SSORPreconditioner;

  const double maxError(1e-10);

  SolverTest<Operator,Vector> solverTest(fop,b0,x0,maxError);

  DummyPreconditioner dummyPrec(1.);

  const FIELD_TYPE relaxFactor(1.);
  JacobiPreconditioner jacobiPrec1(mat,1,relaxFactor);
  JacobiPreconditioner jacobiPrec2(mat,maxIter,relaxFactor);

  GaussSeidelPreconditioner gsPrec1(mat,1,relaxFactor);
  GaussSeidelPreconditioner gsPrec2(mat,maxIter,relaxFactor);

  SORPreconditioner sorPrec1(mat,1,relaxFactor);
  SORPreconditioner sorPrec2(mat,maxIter,relaxFactor);
  SSORPreconditioner ssorPrec1(mat,1,relaxFactor);
  SSORPreconditioner ssorPrec2(mat,maxIter,relaxFactor);

#if HAVE_SUPERLU
  Dune::SuperLU<BCRSMat> solverSuperLU(mat, true);
  std::cout << "SuperLU converged:  "<< solverTest(solverSuperLU) << std::endl <<  std::endl;
#endif

  typedef  Dune::GradientSolver<Vector> GradientSolver;
  GradientSolver solverGradient(fop,dummyPrec, reduction, maxIter, 1);
  std::cout << "GradientSolver with identity preconditioner converged: " << solverTest(solverGradient) << std::endl <<  std::endl;

  typedef  Dune::CGSolver<Vector> CG;
  CG solverCG(fop,dummyPrec, reduction, maxIter, 1);
  std::cout << "CG with identity preconditioner converged: " << solverTest(solverCG) << std::endl <<  std::endl;

  typedef  Dune::BiCGSTABSolver<Vector> BiCG;
  BiCG solverBiCG(fop,dummyPrec, reduction, maxIter, 1);
  std::cout << "BiCGStab with identity preconditioner converged: " <<  solverTest(solverBiCG) << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> JacobiSolver;
  JacobiSolver solverJacobi1(fop,jacobiPrec1,reduction,maxIter,1);
  std::cout << "LoopSolver with a single Jacobi iteration as preconditioner converged: " << solverTest(solverJacobi1)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> JacobiSolver;
  JacobiSolver solverJacobi2(fop,jacobiPrec2,reduction,maxIter,1);
  std::cout << "LoopSolver with multiple Jacobi iteration as preconditioner converged: " << solverTest(solverJacobi2)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> GaussSeidelSolver;
  GaussSeidelSolver solverGaussSeidel1(fop,gsPrec1,reduction,maxIter,1);
  std::cout << "LoopSolver with a single GaussSeidel iteration as preconditioner converged: " << solverTest(solverGaussSeidel1)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> GaussSeidelSolver;
  GaussSeidelSolver solverGaussSeidel2(fop,gsPrec2,reduction,maxIter,1);
  std::cout << "LoopSolver with multiple GaussSeidel iterations as preconditioner converged: " << solverTest(solverGaussSeidel2) << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> SORSolver;
  SORSolver solverSOR1(fop,sorPrec1,reduction,maxIter,1);
  std::cout << "LoopSolver with a single SOR iteration as preconditioner converged: " << solverTest(solverSOR1)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> SORSolver;
  SORSolver solverSOR2(fop,sorPrec2,reduction,maxIter,1);
  std::cout << "LoopSolver with multiple SOR iterations as preconditioner converged: " << solverTest(solverSOR2)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> SSORSolver;
  SSORSolver solverSSOR1(fop,ssorPrec1,reduction,maxIter,1);
  std::cout << "LoopSolver with a single SSOR iteration as preconditioner converged: " << solverTest(solverSOR2)  << std::endl <<  std::endl;

  typedef  Dune::LoopSolver<Vector> SSORSolver;
  SSORSolver solverSSOR2(fop,ssorPrec2,reduction,maxIter,1);
  std::cout << "LoopSolver with multiple SSOR iterations as preconditioner converged: " << solverTest(solverSSOR2)  << std::endl <<  std::endl;

  typedef Dune::MINRESSolver<Vector> MINRES;
  MINRES solverMINRES(fop,dummyPrec, reduction, maxIter, 1);
  std::cout << "MINRES with identity preconditioner converged: " << solverTest(solverMINRES)  << std::endl <<  std::endl;

  typedef Dune::RestartedGMResSolver<Vector> GMRES;
  GMRES solverGMRES(fop,dummyPrec, reduction, maxIter, maxIter*maxIter, 1);
  std::cout << "GMRES with identity preconditioner converged: " << solverTest(solverGMRES)  << std::endl <<  std::endl;

  const int testCount = solverTest.getNumTests();
  const int errorCount = solverTest.getNumFailures();
  std::cout << "Tested " << testCount << " different solvers or preconditioners " << " for a laplacian with complex rhs. " << testCount -  errorCount << " out of " << testCount << " solvers converged! " << std::endl << std::endl;

  std::cout << "============================================" << '\n'
            << "starting solver tests with complex matrices and complex rhs... "
            << std::endl
            << std::endl;

  std::cout << "============================================" << '\n'
            << "solving system with Hilbert matrix of size 10 times imaginary unit... "
            << std::endl
            << std::endl;

  Dune::FieldMatrix<std::complex<double>,10,10> hilbertmatrix;
  for(int i=0; i<10; i++) {
    for(int j=0; j<10; j++) {
      std::complex<double> temp(0.0,1./(i+j+1));
      hilbertmatrix[i][j] = temp;
    }
  }

  Dune::FieldVector<std::complex<double>,10> hilbertsol(1.0);
  Dune::FieldVector<std::complex<double>,10> hilbertiter(0.0);
  Dune::MatrixAdapter<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > hilbertadapter(hilbertmatrix);

  Dune::FieldVector<std::complex<double>,10> hilbertrhs(0.0);
  hilbertadapter.apply(hilbertsol,hilbertrhs);

  Dune::Richardson<Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > noprec(1.0);

  Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > realgmrestest(hilbertadapter,noprec,reduction,maxIter,maxIter,2);
  Dune::InverseOperatorResult stat;
  realgmrestest.apply(hilbertiter,hilbertrhs,stat);

  std::cout << hilbertiter << std::endl;
  // error of solution
  hilbertiter -= hilbertsol;
  std::cout << "error of solution with GMRes:" << std::endl;
  std::cout << hilbertiter.two_norm() << std::endl;

  std::cout << "============================================" << '\n'
            << "solving system with complex matrix of size 10" << '\n'
            << "randomly generated with the Eigen library... "
            << std::endl << std::endl;

  Dune::FieldMatrix<std::complex<double>,10,10> complexmatrix(0.0), hermitianmatrix(0.0);
  Dune::FieldVector<std::complex<double>,10> complexsol, complexrhs(0.0), complexiter(0.0);

  // assemble randomly generated matrices from Eigen
  assemblecomplexmatrix(complexmatrix);
  assemblecomplexsol(complexsol);
  assemblehermitianmatrix(hermitianmatrix);

  Dune::MatrixAdapter<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > complexadapter(complexmatrix), hermitianadapter(hermitianmatrix);

  Dune::SeqJac<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10>,0> complexjacprec(complexmatrix,1,1.0);
  Dune::Richardson<Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > complexnoprec(1.0);

  Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > complexgmrestest(complexadapter,complexnoprec,1e-12,maxIter,maxIter*maxIter,2);

  complexadapter.apply(complexsol,complexrhs);
  complexgmrestest.apply(complexiter,complexrhs,stat);

  std::cout << complexiter << std::endl;
  // error of solution
  complexiter -= complexsol;
  std::cout << "error of solution with GMRes: " << complexiter.two_norm() << std::endl;

  std::cout << "============================================" << '\n'
            << "solving system with hermitian matrix of size 10" << '\n'
            << "randomly generated with the Eigen libary... "
            << std::endl << std::endl;

  Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > hermitiangmrestest(hermitianadapter,complexnoprec,1e-12,maxIter,maxIter*maxIter,2);
  Dune::MINRESSolver<Dune::FieldVector<std::complex<double>,10> > complexminrestest(hermitianadapter,complexnoprec,1e-12,maxIter,2);

  complexiter = 0.0;
  hermitianadapter.apply(complexsol,complexrhs);
  hermitiangmrestest.apply(complexiter,complexrhs,stat);

  std::cout << complexiter << std::endl;
  // error of solution
  complexiter -= complexsol;
  std::cout << "error of solution with GMRes: " << complexiter.two_norm() << std::endl;

  complexiter = 0.0;
  hermitianadapter.apply(complexsol,complexrhs);
  complexminrestest.apply(complexiter,complexrhs,stat);

  std::cout << complexiter << std::endl;
  // error of solution
  complexiter-= complexsol;
  std::cout << "error of solution with MinRes: " << complexiter.two_norm() << std::endl;

  return errorCount;
}
