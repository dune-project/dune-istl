// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/common/parametertree.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solverfactory.hh>

// include direct solvers to see whether it compiles (should throw an exception if used)
#include <dune/istl/cholmod.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/spqr.hh>
#include <dune/istl/paamg/amg.hh>


using namespace Dune;

template<class V>
class MatrixFreeLaplacian2D
  : public LinearOperator<V, V>{

public:
  using domain_type = V;
  using range_type = V;
  using field_type = typename domain_type::field_type;

  MatrixFreeLaplacian2D() = default;

  void apply(const domain_type& x, range_type& y) const override{
    assert(x.N() == y.N());
    y[0] = 2*x[0] - x[1];
    for(size_t i=1; i<y.N()-1; ++i){
      y[i] = -x[i-1] + 2*x[i] -x[i+1];
    }
    y[y.N()-1] = -x[y.N()-2] + 2*x[y.N()-1];
  }

  void applyscaleadd(field_type alpha, const domain_type& x, range_type& y) const override{
    assert(x.N() == y.N());
    y[0] += alpha*(2*x[0] - x[1]);
    for(size_t i=1; i<y.N()-1; ++i){
      y[i] += alpha*(-x[i-1] + 2*x[i] -x[i+1]);
    }
    y[y.N()-1] += alpha*(-x[y.N()-2] + 2*x[y.N()-1]);
  }

  SolverCategory::Category category() const override{
    return SolverCategory::sequential;
  }
};

template<class V, int k>
class MatrixFreeBlockJacobi
  : public Preconditioner<V,V>{
public:
  using domain_type = V;
  using range_type = V;
  using field_type = typename V::field_type;

  MatrixFreeBlockJacobi()
  {
    dia_block_inv = 0.;
    for(size_t i=0;i<k-1; ++i){
      dia_block_inv[i][i] = 2.;
      dia_block_inv[i+1][i] = -1.;
      dia_block_inv[i][i+1] = -1.;
    }
    dia_block_inv[k-1][k-1] = 2.;
    dia_block_inv.invert();
  }

  void pre(domain_type& x, range_type& y) override {}
  void post(domain_type& x) override {}

  void apply(domain_type& x, const range_type& y) override{
    assert(x.N() == y.N());
    size_t n = y.N();
    for(size_t i=0;i<n;++i){
      for(size_t j=0; j<k;++j)
        x[i] += dia_block_inv[i%k][j]*y[k*(i/k)+j];
    }
  }

  SolverCategory::Category category() const override{
    return SolverCategory::sequential;
  }

protected:
  FieldMatrix<field_type, k, k> dia_block_inv;
};

using V = BlockVector<float>;

int main(int argc, char** argv){
  ParameterTree config;
  config["type"] = "minressolver";
  config["verbose"] = "2";
  config["maxit"] = "10000";
  config["reduction"] = "1e-8";
  config.report();
  initSolverFactories<MatrixFreeLaplacian2D<V>>();
  std::shared_ptr<MatrixFreeLaplacian2D<V>> op = std::make_shared<MatrixFreeLaplacian2D<V>>();

  using Prec = MatrixFreeBlockJacobi<V, 10>;
  std::shared_ptr<Prec> prec = std::make_shared<Prec>();

  auto solver = getSolverFromFactory(op, config, prec);
  V b(1000);
  V x = b;
  x = 1.;
  b = 0.;
  InverseOperatorResult res;
  solver->apply(x,b,res);
  return 0;
}
