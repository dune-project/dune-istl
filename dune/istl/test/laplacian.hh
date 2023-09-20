// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef LAPLACIAN_HH
#define LAPLACIAN_HH

#include <optional>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/scalarmatrixview.hh>

template<class B, class Alloc>
void setupSparsityPattern(Dune::BCRSMatrix<B,Alloc>& A, int N)
{
  typedef typename Dune::BCRSMatrix<B,Alloc> Matrix;
  A.setSize(N*N, N*N, N*N*5);
  A.setBuildMode(Matrix::row_wise);

  for (typename Dune::BCRSMatrix<B,Alloc>::CreateIterator i = A.createbegin(); i != A.createend(); ++i) {
    int x = i.index()%N; // x coordinate in the 2d field
    int y = i.index()/N;  // y coordinate in the 2d field

    if(y>0)
      // insert lower neighbour
      i.insert(i.index()-N);
    if(x>0)
      // insert left neighbour
      i.insert(i.index()-1);

    // insert diagonal value
    i.insert(i.index());

    if(x<N-1)
      //insert right neighbour
      i.insert(i.index()+1);
    if(y<N-1)
      // insert upper neighbour
      i.insert(i.index()+N);
  }
}

/* \brief Setup a FD Laplace matrix on a square 2D grid
 *
 * \param A The resulting matrix
 * \param N Number of grid nodes per direction
 * \param reg (optional) Regularization added to the diagonal to ensure a good condition number
 */
template<class B, class Alloc>
void setupLaplacian(Dune::BCRSMatrix<B,Alloc>& A,
                    int N,
                    std::optional<typename Dune::BCRSMatrix<B,Alloc>::field_type> reg = {} )
{
  using field_type = typename Dune::BCRSMatrix<B,Alloc>::field_type;

  setupSparsityPattern(A,N);

  B diagonal(static_cast<field_type>(0)), bone(static_cast<field_type>(0));

  auto setDiagonal = [](auto&& scalarOrMatrix, const auto& value) {
    auto&& matrix = Dune::Impl::asMatrix(scalarOrMatrix);
    for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
      (*rowIt)[rowIt.index()] = value;
  };

  setDiagonal(diagonal, 4.0);
  setDiagonal(bone, -1.0);

  for (auto i = A.begin(); i != A.end(); ++i)
  {
    int x = i.index()%N; // x coordinate in the 2d field
    int y = i.index()/N;  // y coordinate in the 2d field

    i->operator[](i.index())=diagonal;

    if(y>0)
      i->operator[](i.index()-N)=bone;

    if(y<N-1)
      i->operator[](i.index()+N)=bone;

    if(x>0)
      i->operator[](i.index()-1)=bone;

    if(x < N-1)
      i->operator[](i.index()+1)=bone;
  }

  // add regularization to the diagonal
  if ( reg.has_value() )
  {
    for ( auto rowIt=A.begin(); rowIt!=A.end(); rowIt++ )
    {
      for ( auto entryIt=rowIt->begin(); entryIt!=rowIt->end(); entryIt++ )
      {
        if ( entryIt.index() == rowIt.index() )
        {
          auto&& block = Dune::Impl::asMatrix(*entryIt);
          for ( auto it=block.begin(); it!=block.end(); ++it )
          {
            (*it)[it.index()] += reg.value();
          }
        }
      }
    }
  }
}
template<int BS, class Alloc>
void setBoundary(Dune::BlockVector<Dune::FieldVector<double,BS>, Alloc>& lhs,
                 Dune::BlockVector<Dune::FieldVector<double,BS>, Alloc>& rhs,
                 int N)
{
  for(int i=0; i < lhs.size(); ++i) {
    int x = i/N;
    int y = i%N;

    if(x==0 || y ==0 || x==N-1 || y==N-1) {
      lhs[i]=rhs[i]=0;
    }
  }
}
#endif
