#include <config.h>

#include <iostream>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/istl/matrixmarket.hh>
#include <dune/istl/blockkrylov/blockcg.hh>
#include <dune/istl/blockkrylov/blockgmres.hh>
#include <dune/istl/blockkrylov/blockbicgstab.hh>
#include <dune/istl/blockkrylov/utils.hh>

#include <dune/istl/test/laplacian.hh>

using namespace Dune;

using SIMD = LoopSIMD<double, 8, 32>;
using Vector = BlockVector<FieldVector<SIMD, 1>>;
using MatrixBlock = FieldMatrix<double, 1, 1>;
using Mat = BCRSMatrix<MatrixBlock>;
using Operator = MatrixAdapter<Mat, Vector, Vector>;

void test(const std::shared_ptr<Operator>& op, const Vector& b, const ParameterTree& config){
  std::shared_ptr<Preconditioner<Vector,Vector>> prec = SolverFactory<Operator>::getPreconditioner(op,config.sub("preconditioner"));
  std::shared_ptr<InverseOperator<Vector, Vector>> solver = getSolverFromFactory(op, config, prec);
  Dune::InverseOperatorResult res;
  Vector x(b);
  x = 0.0;
  Vector bb(b);

  // manipulate RHS to provoke breakdown
  Vector tmp(b), tmp2(b);
  size_t breakdown_it = config.get<size_t>("breakdown_it", 2);
  size_t breakdown_lanes = config.get<size_t>("breakdown_lanes", 1);
  std::cout << "manipulating RHS" << std::endl;
  for(size_t i=0; i<breakdown_it; ++i){
    tmp = 0.0;
    prec->apply(tmp, tmp2);
    op->apply(tmp, tmp2);
    tmp2 /= tmp2.two_norm();
  }
  for(size_t i=0;i<bb.size();++i){
    for(size_t j=0;j<breakdown_lanes;++j){
      Simd::lane(j+breakdown_lanes, bb[i][0]) = Simd::lane(j, tmp2[i][0]);
    }
  }

  Vector r = bb;
  solver->apply(x,bb,res);
  if(!res.converged)
    DUNE_THROW(Dune::Exception, "not converged!");

  // check residual
  op->applyscaleadd(-1.0, x, r);
  std::cout << r.two_norm() << std::endl;
}

int main(int argc, char** argv){

  ParameterTree config;
  ParameterTreeParser::readINITree("breakdown_test.ini", config);
  ParameterTreeParser::readOptions(argc, argv, config);

  Mat mat;

  std::string matrixfilename = config.get("matrix", "laplacian");
  if(matrixfilename == "laplacian"){
    int N = config.get("N", 10);
    setupLaplacian(mat, N);
  }else{
    loadMatrixMarket(mat, matrixfilename);
  }
  Vector b(mat.N());
  fillRandom(b, true);

  std::shared_ptr<Operator> op = std::make_shared<Operator>(mat);
  initSolverFactories<Operator>();

  for(std::string subkey : config.getSubKeys()){
    test(op, b, config.sub(subkey));
  }
  return 0;
}
