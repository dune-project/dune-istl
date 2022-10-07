// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef ANISOTROPIC_HH
#define  ANISOTROPIC_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/owneroverlapcopy.hh>

typedef Dune::OwnerOverlapCopyAttributeSet GridAttributes;
typedef GridAttributes::AttributeSet GridFlag;
typedef Dune::ParallelLocalIndex<GridFlag> LocalIndex;

template<class M, class G, class L, int n>
void setupPattern(int N, M& mat, Dune::ParallelIndexSet<G,L,n>& indices, int overlapStart, int overlapEnd,
                  int start, int end);

template<class M>
void fillValues(int N, M& mat, int overlapStart, int overlapEnd, int start, int end);


template<class M, class G, class L, int s>
void setupPattern(int N, M& mat, Dune::ParallelIndexSet<G,L,s>& indices, int overlapStart, int overlapEnd,
                  int start, int end)
{
  int n = overlapEnd - overlapStart;

  typename M::CreateIterator iter = mat.createbegin();
  indices.beginResize();

  for(int j=0; j < N; j++)
    for(int i=overlapStart; i < overlapEnd; i++, ++iter) {
      int global = j*N+i;
      GridFlag flag = GridAttributes::owner;
      bool isPublic = false;

      if((i<start && i > 0) || (i>= end && i < N-1))
        flag=GridAttributes::copy;

      if(i<start+1 || i>= end-1) {
        isPublic = true;
        indices.add(global, LocalIndex(iter.index(), flag, isPublic));
      }


      iter.insert(iter.index());

      // i direction
      if(i > overlapStart )
        // We have a left neighbour
        iter.insert(iter.index()-1);

      if(i < overlapEnd-1)
        // We have a rigt neighbour
        iter.insert(iter.index()+1);

      // j direction
      // Overlap is a dirichlet border, discard neighbours
      if(flag != GridAttributes::copy) {
        if(j>0)
          // lower neighbour
          iter.insert(iter.index()-n);
        if(j<N-1)
          // upper neighbour
          iter.insert(iter.index()+n);
      }
    }
  indices.endResize();
}

template<class M, class T>
void fillValues([[maybe_unused]] int N, M& mat, int overlapStart, int overlapEnd, int start, int end, T eps)
{
  typedef typename M::block_type Block;
  Block dval(0), bone(0), bmone(0), beps(0);

  auto setDiagonal = [](auto&& scalarOrMatrix, const auto& value) {
    auto&& matrix = Dune::Impl::asMatrix(scalarOrMatrix);
    for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
      (*rowIt)[rowIt.index()] = value;
  };

  using real_type = typename Dune::FieldTraits<typename M::field_type>::real_type;
  setDiagonal(dval, static_cast<real_type>(2.0)+static_cast<real_type>(2.0)*eps);
  setDiagonal(bone, static_cast<real_type>(1.0));
  setDiagonal(bmone, static_cast<real_type>(-1.0));
  setDiagonal(beps, -eps);

  int n = overlapEnd-overlapStart;
  typedef typename M::ColIterator ColIterator;
  typedef typename M::RowIterator RowIterator;

  for (RowIterator i = mat.begin(); i != mat.end(); ++i) {
    // calculate coordinate
    int y = i.index() / n;
    int x = overlapStart + i.index() - y * n;

    ColIterator endi = (*i).end();

    if(x<start || x >= end) { // || x==0 || x==N-1 || y==0 || y==N-1){
      // overlap node is dirichlet border
      ColIterator j = (*i).begin();

      for(; j.index() < i.index(); ++j)
        *j=0;

      *j = bone;

      for(++j; j != endi; ++j)
        *j=0;
    }else{
      for(ColIterator j = (*i).begin(); j != endi; ++j)
        if(j.index() == i.index())
          *j=dval;
        else if(j.index()+1==i.index() || j.index()-1==i.index())
          *j=beps;
        else
          *j=bmone;
    }
  }
}

template<class V, class G, class L, int s>
void setBoundary(V& lhs, V& rhs, const G& n, Dune::ParallelIndexSet<G,L,s>& indices)
{
  typedef typename Dune::ParallelIndexSet<G,L,s>::const_iterator Iter;
  for(Iter i=indices.begin(); i != indices.end(); ++i) {
    G x = i->global()/n;
    G y = i->global()%n;

    if(x==0 || y ==0 || x==n-1 || y==n-1) {
      //double h = 1.0 / ((double) (n-1));
      lhs[i->local()]=rhs[i->local()]=0; //((double)x)*((double)y)*h*h;
    }
  }
}

template<class V, class G>
void setBoundary(V& lhs, V& rhs, const G& N)
{
  for(int j=0; j < N; ++j)
    for(int i=0; i < N; i++)
      if(i==0 || j ==0 || i==N-1 || j==N-1)
        lhs[j*N+i]=rhs[j*N+i]=0;
}

/**
 * \tparam M A matrix type
 */
template<class MatrixEntry, class G, class L, class C, int s>
Dune::BCRSMatrix<MatrixEntry> setupAnisotropic2d(int N, Dune::ParallelIndexSet<G,L,s>& indices, const Dune::Communication<C>& p, int *nout, typename Dune::BCRSMatrix<MatrixEntry>::field_type eps=1.0)
{
  int procs=p.size(), rank=p.rank();

  using BCRSMat = Dune::BCRSMatrix<MatrixEntry>;

  // calculate size of local matrix in the distributed direction
  int start, end, overlapStart, overlapEnd;

  int n = N/procs; // number of unknowns per process
  int bigger = N%procs; // number of process with n+1 unknows

  // Compute owner region
  if(rank<bigger) {
    start = rank*(n+1);
    end   = start+(n+1);
  }else{
    start = bigger*(n+1) + (rank-bigger) * n;
    end   = start+n;
  }

  // Compute overlap region
  if(start>0)
    overlapStart = start - 1;
  else
    overlapStart = start;

  if(end<N)
    overlapEnd = end + 1;
  else
    overlapEnd = end;

  int noKnown = overlapEnd-overlapStart;

  *nout = noKnown;

  BCRSMat mat(noKnown*N, noKnown*N, noKnown*N*5, BCRSMat::row_wise);

  setupPattern(N, mat, indices, overlapStart, overlapEnd, start, end);
  fillValues(N, mat, overlapStart, overlapEnd, start, end, eps);

  //  Dune::printmatrix(std::cout,mat,"aniso 2d","row",9,1);

  return mat;
}
#endif
