// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#include <config.h>

#include <dune/common/parametertree.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/solverfactory.hh>

#include <dune/istl/umfpack.hh>

namespace Dune {
  DUNE_REGISTER_DIRECT_SOLVER("dummy_direct", [](auto tl, auto op, const ParameterTree& config){
    using TL = decltype(tl);
    using M = typename TypeListElement<0, TL>::type;
    using V = typename TypeListElement<1, TL>::type;
    using U = typename TypeListElement<2, TL>::type;
    return std::make_shared<UMFPack<M>>();
  });
  DUNE_REGISTER_ITERATIVE_SOLVER("dummy_iterative", [](auto tl, auto op, const ParameterTree& config){
    using TL = decltype(tl);
    using V = typename Dune::TypeListElement<0, TL>::type;
    using U = typename Dune::TypeListElement<1, TL>::type;
    std::shared_ptr<Preconditioner<V,U>> preconditioner = getPreconditionerFromFactory(op, config.sub("preconditioner"));
    return std::make_shared<CGSolver<V>>(op, preconditioner, config);
  });
  DUNE_REGISTER_PRECONDITIONER("dummy_prec", [](auto tl, auto op, const ParameterTree& config){
    using TL = decltype(tl);
    using M = typename TypeListElement<0, TL>::type;
    using V = typename TypeListElement<1, TL>::type;
    using U = typename TypeListElement<2, TL>::type;
    return std::make_shared<Richardson<V,U>>();
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
