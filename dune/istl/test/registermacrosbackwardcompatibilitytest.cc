// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>

#include <dune/common/parametertree.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/solverfactory.hh>

#include <dune/istl/umfpack.hh>

namespace Dune {
  DUNE_REGISTER_SOLVER("dummy_direct", [](auto opInfo, auto op, const ParameterTree& config){
    using OpInfo = std::decay_t<decltype(opInfo)>;
    using M = typename OpInfo::matrix_type;
    using U = typename OpInfo::domain_type;
    using V = typename OpInfo::range_type;
    return std::make_shared<UMFPack<M>>();
  });
  DUNE_REGISTER_SOLVER("dummy_iterative", [](auto opInfo, auto op, const ParameterTree& config){
    using OpInfo = std::decay_t<decltype(opInfo)>;
    using U = typename OpInfo::domain_type;
    using V = typename OpInfo::range_type;
    std::shared_ptr<Preconditioner<V,U>> preconditioner = getPreconditionerFromFactory(op, config.sub("preconditioner"));
    return std::make_shared<CGSolver<V>>(op, preconditioner, config);
  });
  DUNE_REGISTER_PRECONDITIONER("dummy_prec", [](auto opInfo, auto op, const ParameterTree& config){
    using OpInfo = std::decay_t<decltype(opInfo)>;
    using M = typename OpInfo::matrix_type;
    using U = typename OpInfo::domain_type;
    using V = typename OpInfo::range_type;
    return std::make_shared<Richardson<U,V>>();
  });
}

using namespace Dune;

int main(){
  using Mat = BCRSMatrix<double>;
  using V = BlockVector<double>;
  using Op = MatrixAdapter<Mat, V, V>;
  Mat A;
  std::shared_ptr<Op> op = std::make_shared<Op>(A);

  initSolverFactories<Op>();
  ParameterTree config_direct;
  config_direct["type"] = "dummy_direct";
  auto direct_solver = getSolverFromFactory(op, config_direct);

  ParameterTree config_iterative;
  config_iterative["type"] = "dummy_iterative";
  config_iterative["verbose"] = "1";
  config_iterative["maxit"] = "100";
  config_iterative["reduction"] = "1e-6";
  config_iterative.sub("preconditioner")["type"] = "dummy_prec";
  auto iterative_solver = getSolverFromFactory(op, config_iterative);
  return 0;
}
