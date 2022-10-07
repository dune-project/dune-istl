// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_GRAPHCREATOR_HH
#define DUNE_AMG_GRAPHCREATOR_HH

#include <tuple>

#include "graph.hh"
#include "dependency.hh"
#include "pinfo.hh"
#include <dune/istl/operators.hh>
#include <dune/istl/bcrsmatrix.hh>

namespace Dune
{
  namespace Amg
  {
    template<class M, class PI>
    struct PropertiesGraphCreator
    {
      typedef typename M::matrix_type Matrix;
      typedef Dune::Amg::MatrixGraph<const Matrix> MatrixGraph;
      typedef Dune::Amg::SubGraph<MatrixGraph,
          std::vector<bool> > SubGraph;
      typedef Dune::Amg::PropertiesGraph<SubGraph,
          VertexProperties,
          EdgeProperties,
          IdentityMap,
          typename SubGraph::EdgeIndexMap>
      PropertiesGraph;

      typedef std::tuple<MatrixGraph*,PropertiesGraph*,SubGraph*> GraphTuple;

      template<class OF, class T>
      static GraphTuple create(const M& matrix, T& excluded,
                               PI& pinfo, const OF& of)
      {
        MatrixGraph* mg = new MatrixGraph(matrix.getmat());
        typedef typename PI::ParallelIndexSet ParallelIndexSet;
        typedef typename ParallelIndexSet::const_iterator IndexIterator;
        IndexIterator iend = pinfo.indexSet().end();

        for(IndexIterator index = pinfo.indexSet().begin(); index != iend; ++index)
          excluded[index->local()] = of.contains(index->local().attribute());

        SubGraph* sg= new SubGraph(*mg, excluded);
        PropertiesGraph* pg = new PropertiesGraph(*sg, IdentityMap(), sg->getEdgeIndexMap());
        return GraphTuple(mg,pg,sg);
      }

      static void free(GraphTuple& graphs)
      {
        delete std::get<2>(graphs);
        delete std::get<1>(graphs);
      }
    };

    template<class M>
    struct PropertiesGraphCreator<M,SequentialInformation>
    {
      typedef typename M::matrix_type Matrix;

      typedef Dune::Amg::MatrixGraph<const Matrix> MatrixGraph;

      typedef Dune::Amg::PropertiesGraph<MatrixGraph,
          VertexProperties,
          EdgeProperties,
          IdentityMap,
          IdentityMap> PropertiesGraph;

      typedef std::tuple<MatrixGraph*,PropertiesGraph*> GraphTuple;

      template<class OF, class T>
      static GraphTuple create([[maybe_unused]] const M& matrix,
                               [[maybe_unused]] T& excluded,
                               [[maybe_unused]] const SequentialInformation& pinfo,
                               const OF&)
      {
        MatrixGraph* mg = new MatrixGraph(matrix.getmat());
        PropertiesGraph* pg = new PropertiesGraph(*mg, IdentityMap(), IdentityMap());
        return GraphTuple(mg,pg);
      }

      static void free(GraphTuple& graphs)
      {
        delete std::get<1>(graphs);
      }

    };

  } //namespace Amg
} // namespace Dune
#endif
