// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef LAPLACIAN_HH
#define LAPLACIAN_HH
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fvector.hh>

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


template<class B, class Alloc>
void setupLaplacian(Dune::BCRSMatrix<B,Alloc>& A, int N)
{
  typedef typename Dune::BCRSMatrix<B,Alloc>::field_type FieldType;

  setupSparsityPattern(A,N);

  B diagonal(static_cast<FieldType>(0)), bone(static_cast<FieldType>(0));

  Dune::Hybrid::ifElse(Dune::IsNumber<B>(),
    [&](auto id) {
      diagonal = B(4.0);
      bone = B(-1.0);
    },
    [&](auto id) {
      for (auto b = id(diagonal).begin(); b != id(diagonal).end(); ++b)
        b->operator[](b.index())=4;

      for (auto b=id(bone).begin(); b != id(bone).end(); ++b)
        b->operator[](b.index())=-1.0;
    });

  for (typename Dune::BCRSMatrix<B,Alloc>::RowIterator i = A.begin(); i != A.end(); ++i) {
    int x = i.index()%N; // x coordinate in the 2d field
    int y = i.index()/N;  // y coordinate in the 2d field

    /*    if(x==0 || x==N-1 || y==0||y==N-1){

       i->operator[](i.index())=1.0;

       if(y>0)
       i->operator[](i.index()-N)=0;

       if(y<N-1)
       i->operator[](i.index()+N)=0.0;

       if(x>0)
       i->operator[](i.index()-1)=0.0;

       if(x < N-1)
       i->operator[](i.index()+1)=0.0;

       }else*/
    {

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
